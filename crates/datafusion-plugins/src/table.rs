//! A [`PlanNode`] as a table provider.
//!
//! [`NodeTable`] plans the node's inputs when it is scanned, under the
//! scanning session's configuration, prunes them to the columns the scan
//! needs, builds the node, and hands the plan out in a [`BridgeExec`]. The
//! inputs come from [`NodeInputs`]:
//!
//! * [`PlanInputs`]: serialized logical plans (datafusion-proto; how
//!   DataFrames from another DataFusion instance, such as datafusion-python
//!   across its FFI boundary, arrive), decoded and planned here. Filter and
//!   projection pushdown into their scans, and their declared orderings,
//!   are visible to the node. Only plans whose scans read local files
//!   qualify.
//! * [`crate::HostInputs`]: inputs another engine executes, read as Arrow
//!   streams (anything: in-memory tables, UDFs, object stores).

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, DataFusionError, Result, plan_datafusion_err, plan_err};
use datafusion::datasource::TableType;
use datafusion::datasource::listing::ListingTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionContext;
use datafusion::execution::memory_pool::MemoryPool;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, TableProviderFilterPushDown};
use datafusion::physical_plan::{ExecutionPlan, displayable};
use datafusion::prelude::SessionConfig;

use crate::PlanNode;
use crate::bridge::BridgeExec;

/// The inputs of a [`NodeTable`].
#[async_trait]
pub trait NodeInputs: fmt::Debug + Send + Sync + 'static {
    /// The inputs' full schemas.
    fn schemas(&self) -> &[SchemaRef];

    /// The inputs' physical plans under `config`, each keeping only
    /// `columns` (in schema order; `None`: all).
    async fn plan(&self, config: &ConfigOptions, columns: Vec<Option<Vec<String>>>) -> Result<Vec<Arc<dyn ExecutionPlan>>>;
}

/// Where a [`NodeTable`]'s plans get their memory pool (`None`: the caller's).
type PoolSource = Arc<dyn Fn() -> Option<Arc<dyn MemoryPool>> + Send + Sync>;

/// A [`PlanNode`] over [`NodeInputs`], as a table provider.
#[derive(Clone)]
pub struct NodeTable {
    inputs: Arc<dyn NodeInputs>,
    node: Arc<dyn PlanNode>,
    schema: SchemaRef,
    runtime: Option<tokio::runtime::Handle>,
    pool: Option<PoolSource>,
    last_plan: Arc<Mutex<Option<String>>>,
}

impl fmt::Debug for NodeTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeTable").field("inputs", &self.inputs).field("node", &self.node).finish()
    }
}

impl NodeTable {
    /// Validate `node` against the inputs' schemas.
    pub fn try_new(inputs: Arc<dyn NodeInputs>, node: Arc<dyn PlanNode>) -> Result<Self> {
        let schema = node.schema(inputs.schemas())?;
        Ok(Self { inputs, node, schema, runtime: None, pool: None, last_plan: Arc::default() })
    }

    /// Plan, execute and poll on `runtime` (needed when the caller lives in
    /// another shared library with its own copy of tokio).
    pub fn with_runtime(mut self, runtime: tokio::runtime::Handle) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Run the node's plans with the memory pool `pool` returns when the
    /// table is scanned (`None`: the caller's). DataFusion 55's FFI hands a
    /// plugin a default, unbounded pool.
    pub fn with_memory_pool(mut self, pool: impl Fn() -> Option<Arc<dyn MemoryPool>> + Send + Sync + 'static) -> Self {
        self.pool = Some(Arc::new(pool));
        self
    }

    /// The physical plan of the latest scan (indented), for debugging.
    pub fn last_plan(&self) -> Option<String> {
        self.last_plan.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Plan the inputs under `config` and build the node's plan for the
    /// output columns `projection`.
    pub async fn plan(&self, config: &ConfigOptions, projection: Option<&Vec<usize>>) -> Result<Arc<dyn ExecutionPlan>> {
        let all: Vec<usize> = (0..self.schema.fields().len()).collect();
        let projection = projection.unwrap_or(&all);
        let schemas = self.inputs.schemas();
        let columns = self.node.required_columns(schemas, projection)?;
        let inputs = self.inputs.plan(config, columns).await?;
        let inner = self.node.build(schemas, inputs, projection, config)?;
        let shown = displayable(inner.as_ref()).indent(true).to_string();
        *self.last_plan.lock().unwrap_or_else(PoisonError::into_inner) = Some(shown);
        let pool = self.pool.as_ref().and_then(|p| p());
        Ok(Arc::new(BridgeExec::new(inner, self.runtime.clone(), pool)))
    }
}

#[async_trait]
impl TableProvider for NodeTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(&self, filters: &[&Expr]) -> Result<Vec<TableProviderFilterPushDown>> {
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
            // Plan on our own runtime: the caller may be another library
            // (with its own tokio) polling this future on its threads.
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

/// Serialized logical plans (datafusion-proto) whose scans read local files.
#[derive(Debug)]
pub struct PlanInputs {
    bytes: Vec<Vec<u8>>,
    schemas: Vec<SchemaRef>,
}

impl PlanInputs {
    /// Decode the plans (for their schemas). An error means this route
    /// doesn't apply: a plan doesn't decode, or a scan isn't a local file.
    pub fn try_new(bytes: Vec<Vec<u8>>) -> Result<Self> {
        let ctx = context_for(&ConfigOptions::default(), 1);
        let schemas = bytes
            .iter()
            .map(|b| {
                let plan = decode_plan(b, &ctx)?;
                local_files_only(&plan)?;
                Ok(Arc::new(plan.schema().as_arrow().clone()))
            })
            .collect::<Result<_>>()?;
        Ok(Self { bytes, schemas })
    }
}

#[async_trait]
impl NodeInputs for PlanInputs {
    fn schemas(&self) -> &[SchemaRef] {
        &self.schemas
    }

    async fn plan(&self, config: &ConfigOptions, columns: Vec<Option<Vec<String>>>) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
        let mut plans = Vec::with_capacity(self.bytes.len());
        for (bytes, cols) in self.bytes.iter().zip(columns) {
            let mut lp = decode_plan(bytes, &context_for(config, 1))?;
            if let Some(cols) = cols {
                lp = project(lp, &cols)?;
            }
            // Scans (under filters and projections only) are split into more,
            // smaller byte ranges: the node's merge then decodes several
            // disjoint ranges ahead in parallel (see `merge`). Other plans
            // keep the session's partitioning.
            let split = if scans_only(&lp)? { SCAN_SPLIT } else { 1 };
            plans.push(context_for(config, split).state().create_physical_plan(&lp).await?);
        }
        Ok(plans)
    }
}

/// Decode a datafusion-proto logical plan in `ctx`.
fn decode_plan(bytes: &[u8], ctx: &SessionContext) -> Result<LogicalPlan> {
    datafusion_proto::bytes::logical_plan_from_bytes(bytes, ctx.task_ctx().as_ref())
        .map_err(|e| plan_datafusion_err!("cannot decode an input plan: {e}"))
}

/// Scan ranges per thread for inputs that are only scans.
const SCAN_SPLIT: usize = 8;

/// A context planning like `config`, with `split` times the partitions.
fn context_for(config: &ConfigOptions, split: usize) -> SessionContext {
    // Without work stealing, each scan partition is one byte range read in
    // order, so a sorted file's partitions are each sorted.
    let mut config = config.clone();
    config.execution.enable_file_stream_work_stealing = false;
    config.execution.target_partitions = config.execution.target_partitions.max(1) * split;
    let state = SessionStateBuilder::new().with_config(SessionConfig::from(config)).with_default_features().build();
    SessionContext::new_with_state(state)
}

/// Every scan of `plan` reads local files through a `ListingTable`, so it
/// can be planned here with the same result as in the host session.
fn local_files_only(plan: &LogicalPlan) -> Result<()> {
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            let provider = datafusion::datasource::source_as_provider(&scan.source)?;
            let any: &dyn std::any::Any = provider.as_ref();
            let local = any
                .downcast_ref::<ListingTable>()
                .is_some_and(|t| t.table_paths().iter().all(|u| u.scheme() == "file"));
            if !local {
                return plan_err!("input scan of {} is not a local file table", scan.table_name);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(())
}

/// `plan` has only scans, filters and projections.
fn scans_only(plan: &LogicalPlan) -> Result<bool> {
    let other = plan.exists(|n| {
        Ok(!matches!(n, LogicalPlan::TableScan(_) | LogicalPlan::Filter(_) | LogicalPlan::Projection(_) | LogicalPlan::SubqueryAlias(_)))
    })?;
    Ok(!other)
}

/// Keep only `cols` of `plan`, so the optimizer prunes the input scans.
fn project(plan: LogicalPlan, cols: &[String]) -> Result<LogicalPlan> {
    let schema = Arc::clone(plan.schema());
    let exprs = cols
        .iter()
        .map(|c| {
            let i = schema
                .fields()
                .iter()
                .position(|f| f.name() == c)
                .ok_or_else(|| plan_datafusion_err!("column '{c}' not found"))?;
            Ok(Expr::Column(Column::from(schema.qualified_field(i))))
        })
        .collect::<Result<Vec<_>>>()?;
    LogicalPlanBuilder::from(plan).project(exprs)?.build()
}
