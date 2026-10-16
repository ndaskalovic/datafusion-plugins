//! Inputs executed by another engine, consumed as streams of Arrow batches.
//!
//! The Python package runs every input DataFrame in datafusion-python (on
//! its own runtime, with everything it supports: in-memory tables, Python
//! UDFs, object stores) and hands the node one stream per partition.
//! [`HostInputExec`] exposes those streams as the partitions of a leaf
//! plan; each is opened and read exactly once, on a blocking thread (the
//! host may block on its own runtime, or plan and run other nodes). Their
//! order is not visible here: it comes from the contract (each partition
//! sorted), or the node sorts (when not `assume_sorted`).

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, exec_err, internal_err};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchReceiverStreamBuilder;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties, SendableRecordBatchStream,
};

use crate::table::NodeInputs;

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

/// [`PartitionSource`]s as the inputs of a [`crate::NodeTable`].
#[derive(Debug)]
pub struct HostInputs {
    sources: Vec<Arc<dyn PartitionSource>>,
    schemas: Vec<SchemaRef>,
}

impl HostInputs {
    pub fn new(sources: Vec<Arc<dyn PartitionSource>>) -> Self {
        let schemas = sources.iter().map(|s| s.schema()).collect();
        Self { sources, schemas }
    }
}

#[async_trait]
impl NodeInputs for HostInputs {
    fn schemas(&self) -> &[SchemaRef] {
        &self.schemas
    }

    async fn plan(&self, _config: &ConfigOptions, columns: Vec<Option<Vec<String>>>) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
        let (sources, schemas) = (self.sources.clone(), self.schemas.clone());
        let prepared = tokio::task::spawn_blocking(move || {
            sources
                .iter()
                .zip(schemas.iter().zip(columns))
                .map(|(s, (schema, cols))| {
                    let cols = cols.unwrap_or_else(|| schema.fields().iter().map(|f| f.name().clone()).collect());
                    s.prepare(&cols)
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))??;
        Ok(prepared.into_iter().map(|p| Arc::new(HostInputExec::new(p)) as Arc<dyn ExecutionPlan>).collect())
    }
}

/// The partitions' readers once opened, or why opening failed.
type Readers = Mutex<Option<std::result::Result<Vec<Option<BatchReader>>, Arc<DataFusionError>>>>;

/// The partitions of a [`PreparedSource`] as a leaf plan.
pub(crate) struct HostInputExec {
    src: Arc<dyn PreparedSource>,
    readers: Arc<Readers>,
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
        Self { src, readers: Arc::default(), cache: Arc::new(cache) }
    }
}

/// Partition `i`'s reader; the first call opens every partition. Blocking.
fn take(src: &dyn PreparedSource, readers: &Readers, i: usize) -> Result<Option<BatchReader>> {
    let mut g = readers.lock().unwrap_or_else(PoisonError::into_inner);
    if g.is_none() {
        let n = src.partition_count().max(1);
        // Opened once: a failure is the answer for every partition.
        let opened = src.open().and_then(|rs| match rs.len() > n {
            true => exec_err!("HostInputExec: input has {} partitions, expected {n}", rs.len()),
            false => Ok(rs.into_iter().map(Some).collect()),
        });
        *g = Some(opened.map_err(Arc::new));
    }
    let slots = match g.as_mut().expect("opened") {
        Ok(slots) => slots,
        Err(e) => return Err(DataFusionError::Shared(Arc::clone(e))),
    };
    match slots.get_mut(i) {
        Some(slot) => match slot.take() {
            Some(r) => Ok(Some(r)),
            None => exec_err!(
                "HostInputExec: partition {i} was already executed; a host input is a stream read once \
                 (plan the query again to run it again)"
            ),
        },
        None => Ok(None), // fewer partitions than planned: this one is empty
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

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("HostInputExec has no children");
        }
        Ok(self)
    }

    fn execute(&self, partition: usize, _ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let mut builder = RecordBatchReceiverStreamBuilder::new(self.src.schema(), 2);
        let tx = builder.tx();
        let (src, readers) = (Arc::clone(&self.src), Arc::clone(&self.readers));
        builder.spawn_blocking(move || {
            if let Some(reader) = take(src.as_ref(), &readers, partition)? {
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
