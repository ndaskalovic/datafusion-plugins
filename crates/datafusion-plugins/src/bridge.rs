//! The runtime-bridging leaf every node's plan is handed out in.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Result, internal_err};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::MemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream, SendableRecordBatchStream,
    displayable,
};
use futures::Stream;

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
/// library with its own copy of tokio). If `pool` is set, the inner plan
/// runs with it instead of the caller's memory pool.
pub(crate) struct BridgeExec {
    inner: Arc<dyn ExecutionPlan>,
    runtime: Option<tokio::runtime::Handle>,
    pool: Option<Arc<dyn MemoryPool>>,
    cache: Arc<PlanProperties>,
}

impl fmt::Debug for BridgeExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BridgeExec").field("inner", &self.inner.name()).finish()
    }
}

impl BridgeExec {
    pub fn new(
        inner: Arc<dyn ExecutionPlan>,
        runtime: Option<tokio::runtime::Handle>,
        pool: Option<Arc<dyn MemoryPool>>,
    ) -> Self {
        let p = inner.properties();
        let cache = PlanProperties::new(
            EquivalenceProperties::clone(p.equivalence_properties()),
            p.output_partitioning().clone(),
            p.emission_type,
            p.boundedness,
        );
        Self { inner, runtime, pool, cache: Arc::new(cache) }
    }

    /// `context` with this plan's memory pool, if any.
    fn context(&self, context: Arc<TaskContext>) -> Result<Arc<TaskContext>> {
        let Some(pool) = &self.pool else {
            return Ok(context);
        };
        let runtime = RuntimeEnvBuilder::from_runtime_env(context.runtime_env().as_ref())
            .with_memory_pool(Arc::clone(pool))
            .build_arc()?;
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
}

impl DisplayAs for BridgeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::TreeRender => write!(f, "BridgeExec"),
            _ => {
                let inner = displayable(self.inner.as_ref()).indent(false).to_string();
                write!(f, "BridgeExec: [{}]", inner.trim_end().replace('\n', " <- "))
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

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("BridgeExec has no children");
        }
        Ok(self)
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let context = self.context(context)?;
        match &self.runtime {
            None => self.inner.execute(partition, context),
            Some(rt) => {
                let _g = rt.enter();
                let inner = self.inner.execute(partition, context)?;
                Ok(Box::pin(InRuntime { inner: Some(inner), rt: rt.clone() }))
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
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

    fn poll_next(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        let _g = self.rt.enter();
        match self.inner.as_mut() {
            Some(s) => s.as_mut().poll_next(cx),
            None => std::task::Poll::Ready(None),
        }
    }
}

impl RecordBatchStream for InRuntime {
    fn schema(&self) -> SchemaRef {
        self.inner.as_ref().map(|s| s.schema()).unwrap_or_else(|| Arc::new(Schema::empty()))
    }
}

impl Drop for InRuntime {
    fn drop(&mut self) {
        let _g = self.rt.enter();
        self.inner = None;
    }
}
