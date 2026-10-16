//! The runtime-bridging leaf every node's plan is handed out in.

use std::fmt;
use std::sync::{Arc, Mutex};

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Result, plan_err};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::MemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    RecordBatchStream, ReplaceChildrenOptions, SendableRecordBatchStream, displayable,
};
use futures::Stream;

static LAST_PLAN: Mutex<Option<String>> = Mutex::new(None);
static MEMORY_POOL: Mutex<Option<Arc<dyn MemoryPool>>> = Mutex::new(None);

/// The memory pool for plans run through [`BridgeExec`], or `None` to use
/// the caller's. A caller in another library (datafusion-python over FFI)
/// can't share its pool: DataFusion 55's FFI task context arrives with a
/// default, unbounded one. Set a limit here instead.
pub fn set_memory_pool(pool: Option<Arc<dyn MemoryPool>>) {
    if let Ok(mut g) = MEMORY_POOL.lock() {
        *g = pool;
    }
}

/// `context` with the pool set by [`set_memory_pool`], if any.
fn with_pool(context: Arc<TaskContext>) -> Result<Arc<TaskContext>> {
    let Some(pool) = MEMORY_POOL.lock().ok().and_then(|g| g.clone()) else {
        return Ok(context);
    };
    let runtime = RuntimeEnvBuilder::new().with_memory_pool(pool).build_arc()?;
    Ok(Arc::new(TaskContext::new(
        context.task_id(),
        context.session_id(),
        context.session_config().clone(),
        context.scalar_functions().clone(),
        context.higher_order_functions().clone(),
        context.aggregate_functions().clone(),
        context.window_functions().clone(),
        runtime,
    )))
}

/// The physical plan most recently built by a node's table provider in this process
/// (indented), for debugging and tests.
pub fn last_plan() -> Option<String> {
    LAST_PLAN.lock().ok().and_then(|g| g.clone())
}

pub(crate) fn set_last_plan(plan: &Arc<dyn ExecutionPlan>) {
    let s = displayable(plan.as_ref()).indent(true).to_string();
    if let Ok(mut g) = LAST_PLAN.lock() {
        *g = Some(s);
    }
}

/// Leaf node holding a node's finished plan.
///
/// A plan handed to another session (over FFI, or through a table provider)
/// is optimized again by that session, and DataFusion's distribution and
/// sorting rules would see the scans below the join and repartition or merge
/// them. As a leaf with the inner plan's properties (one ordered partition)
/// it is left alone; the inner plan is already in final form.
///
/// If `runtime` is set, `execute` and every poll of the output stream run
/// inside that tokio runtime (needed when the caller lives in another shared
/// library with its own copy of tokio).
pub struct BridgeExec {
    inner: Arc<dyn ExecutionPlan>,
    runtime: Option<tokio::runtime::Handle>,
    cache: Arc<PlanProperties>,
}

impl fmt::Debug for BridgeExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BridgeExec").field("inner", &self.inner.name()).finish()
    }
}

impl BridgeExec {
    pub fn new(inner: Arc<dyn ExecutionPlan>, runtime: Option<tokio::runtime::Handle>) -> Self {
        let p = inner.properties();
        let cache = PlanProperties::new(
            EquivalenceProperties::clone(p.equivalence_properties()),
            p.output_partitioning().clone(),
            p.emission_type,
            p.boundedness,
        );
        Self {
            inner,
            runtime,
            cache: Arc::new(cache),
        }
    }

    pub fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
}

impl DisplayAs for BridgeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::TreeRender => write!(f, "BridgeExec"),
            _ => {
                let inner = displayable(self.inner.as_ref()).indent(false).to_string();
                let inner = inner.trim_end().replace('\n', " <- ");
                write!(f, "BridgeExec: [{inner}]")
            }
        }
    }
}

impl ExecutionPlan for BridgeExec {
    fn name(&self) -> &str {
        "BridgeExec"
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
            return plan_err!("BridgeExec has no children");
        }
        Ok(self)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return plan_err!("BridgeExec has no children");
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let context = with_pool(context)?;
        match &self.runtime {
            None => self.inner.execute(partition, context),
            Some(rt) => {
                let _g = rt.enter();
                let inner = self.inner.execute(partition, context)?;
                Ok(Box::pin(InRuntime {
                    inner: Some(inner),
                    rt: rt.clone(),
                }))
            }
        }
    }

    fn metrics(&self) -> Option<datafusion::physical_plan::metrics::MetricsSet> {
        self.inner.metrics()
    }
}

/// Polls (and drops) `inner` inside `rt`.
struct InRuntime {
    inner: Option<SendableRecordBatchStream>,
    rt: tokio::runtime::Handle,
}

impl Stream for InRuntime {
    type Item = Result<arrow::array::RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let _g = self.rt.enter();
        match self.inner.as_mut() {
            Some(s) => s.as_mut().poll_next(cx),
            None => std::task::Poll::Ready(None),
        }
    }
}

impl RecordBatchStream for InRuntime {
    fn schema(&self) -> SchemaRef {
        self.inner
            .as_ref()
            .map(|s| s.schema())
            .unwrap_or_else(|| Arc::new(Schema::empty()))
    }
}

impl Drop for InRuntime {
    fn drop(&mut self) {
        let _g = self.rt.enter();
        self.inner = None;
    }
}
