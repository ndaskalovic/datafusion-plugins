//! Inputs executed by another engine, consumed as streams of Arrow batches.
//!
//! The Python package runs every input DataFrame in datafusion-python (on
//! its own runtime, with everything it supports: in-memory tables, Python
//! UDFs, object stores) and hands the node one stream per partition.
//! [`HostInputExec`] exposes those streams as the partitions of a leaf
//! plan; each is read exactly once, on a blocking thread. Their order is not
//! visible here: it comes from the contract (each partition sorted), or the
//! node sorts (when not `assume_sorted`).

use std::fmt;
use std::sync::{Arc, Mutex};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, exec_err, internal_datafusion_err, plan_err};
use datafusion::datasource::TableType;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchReceiverStreamBuilder;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    ReplaceChildrenOptions, SendableRecordBatchStream,
};

use crate::bridge::{BridgeExec, set_last_plan};
use crate::plan_bytes::PlanNode;

/// A blocking reader of one partition's batches.
pub type BatchReader = Box<dyn Iterator<Item = Result<RecordBatch>> + Send>;

/// An input another engine executes.
pub trait PartitionSource: fmt::Debug + Send + Sync + 'static {
    fn schema(&self) -> SchemaRef;
    /// Keep only `columns` (in schema order) and fix the partitioning.
    /// Blocking.
    fn prepare(&self, columns: &[String]) -> Result<Arc<dyn PreparedSource>>;
}

/// A prepared input: fixed schema and partition count.
pub trait PreparedSource: fmt::Debug + Send + Sync + 'static {
    fn schema(&self) -> SchemaRef;
    fn partition_count(&self) -> usize;
    /// Start every partition (once); one reader per partition. Blocking.
    fn open(&self) -> Result<Vec<BatchReader>>;
}

/// The partitions of a [`PreparedSource`] as a leaf plan.
pub struct HostInputExec {
    src: Arc<dyn PreparedSource>,
    readers: Mutex<Option<Vec<Option<BatchReader>>>>,
    cache: Arc<PlanProperties>,
}

impl fmt::Debug for HostInputExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostInputExec").field("src", &self.src).finish()
    }
}

impl HostInputExec {
    pub fn new(src: Arc<dyn PreparedSource>) -> Self {
        let cache = PlanProperties::new(
            EquivalenceProperties::new(src.schema()),
            Partitioning::UnknownPartitioning(src.partition_count().max(1)),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            src,
            readers: Mutex::new(None),
            cache: Arc::new(cache),
        }
    }

    /// Partition `i`'s reader; the first call opens every partition.
    fn take(&self, i: usize) -> Result<Option<BatchReader>> {
        let mut g = self.readers.lock().map_err(|_| internal_datafusion_err!("poisoned lock"))?;
        if g.is_none() {
            let rs = self.src.open()?;
            let n = self.src.partition_count().max(1);
            if rs.len() > n {
                return exec_err!("input has {} partitions, expected {n}", rs.len());
            }
            *g = Some(rs.into_iter().map(Some).collect());
        }
        let slots = g.as_mut().expect("opened");
        match slots.get_mut(i) {
            Some(slot) => slot
                .take()
                .map(Some)
                .ok_or_else(|| internal_datafusion_err!("input partition {i} executed twice")),
            None => Ok(None), // fewer partitions than planned: this one is empty
        }
    }
}

impl DisplayAs for HostInputExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "HostInputExec: partitions={}", self.src.partition_count())
    }
}

impl ExecutionPlan for HostInputExec {
    fn name(&self) -> &str {
        "HostInputExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return plan_err!("HostInputExec has no children");
        }
        Ok(self)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(datafusion::physical_plan::ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(&self, partition: usize, _ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let mut builder = RecordBatchReceiverStreamBuilder::new(self.src.schema(), 2);
        let tx = builder.tx();
        let reader = self.take(partition)?;
        builder.spawn_blocking(move || {
            if let Some(reader) = reader {
                for b in reader {
                    let failed = b.is_err();
                    if tx.blocking_send(b).is_err() || failed {
                        break;
                    }
                }
            }
            Ok(())
        });
        Ok(builder.build())
    }
}

/// A [`PlanNode`] over inputs another engine executes, as a table provider.
#[derive(Debug, Clone)]
pub struct StreamInputTable {
    sources: Vec<Arc<dyn PartitionSource>>,
    schemas: Vec<SchemaRef>,
    node: Arc<dyn PlanNode>,
    schema: SchemaRef,
    runtime: Option<tokio::runtime::Handle>,
}

impl StreamInputTable {
    pub fn try_new(
        sources: Vec<Arc<dyn PartitionSource>>,
        node: Arc<dyn PlanNode>,
        runtime: Option<tokio::runtime::Handle>,
    ) -> Result<Self> {
        let schemas: Vec<SchemaRef> = sources.iter().map(|s| s.schema()).collect();
        let schema = node.schema(&schemas)?;
        Ok(Self {
            sources,
            schemas,
            node,
            schema,
            runtime,
        })
    }

    /// Prepare the inputs and build the node (blocking calls into the
    /// sources happen on a blocking thread).
    pub async fn plan(
        &self,
        config: &ConfigOptions,
        projection: Option<&Vec<usize>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let all: Vec<usize> = (0..self.schema.fields().len()).collect();
        let projection: Vec<usize> = projection.cloned().unwrap_or(all);
        let needed = self.node.required_columns(&self.schemas, &projection)?;
        let (sources, schemas) = (self.sources.clone(), self.schemas.clone());
        let prepared = tokio::task::spawn_blocking(move || {
            sources
                .iter()
                .zip(schemas.iter().zip(needed))
                .map(|(s, (schema, cols))| {
                    let cols = cols.unwrap_or_else(|| {
                        schema.fields().iter().map(|f| f.name().clone()).collect()
                    });
                    s.prepare(&cols)
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;
        let inputs: Vec<Arc<dyn ExecutionPlan>> = prepared
            .into_iter()
            .map(|p| Arc::new(HostInputExec::new(p)) as Arc<dyn ExecutionPlan>)
            .collect();
        let inner = self.node.build(&self.schemas, inputs, &projection, config)?;
        set_last_plan(&inner);
        Ok(Arc::new(BridgeExec::new(inner, self.runtime.clone())))
    }
}

#[async_trait]
impl TableProvider for StreamInputTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Unsupported; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let config = state.config_options().clone();
        let projection = projection.cloned();
        match &self.runtime {
            Some(rt) => {
                let this = self.clone();
                rt.spawn(async move { this.plan(&config, projection.as_ref()).await })
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?
            }
            None => self.plan(&config, projection.as_ref()).await,
        }
    }
}
