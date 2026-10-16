//! A node over serialized logical plans, as a table provider.
//!
//! This is how DataFrames from another DataFusion instance (for example
//! datafusion-python, across its FFI boundary) become a node's inputs: the
//! caller serializes each input's logical plan with datafusion-proto
//! (`DataFrame.logical_plan().to_bytes()` in Python), and [`PlanBytesTable`]
//! decodes and plans them in its own `SessionContext`, configured from the
//! session that scans it, then hands the physical inputs to a [`PlanNode`].
//! Planning locally means filter and projection pushdown into the input
//! scans, and their declared orderings, are fully visible to the node.
//!
//! Only plans datafusion-proto can serialize work: file scans (Parquet, CSV,
//! JSON, ...) under any standard operators. In-memory tables and Python UDFs
//! cannot be serialized on DataFusion 55. Files are read from the local
//! filesystem.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::config::ConfigOptions;
use datafusion::common::{Column, Result, plan_datafusion_err};
use datafusion::datasource::TableType;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionConfig;

use crate::bridge::{BridgeExec, set_last_plan};

/// Decode a datafusion-proto logical plan in `ctx`.
pub(crate) fn decode_plan(bytes: &[u8], ctx: &SessionContext) -> Result<LogicalPlan> {
    datafusion_proto::bytes::logical_plan_from_bytes(bytes, ctx.task_ctx().as_ref())
        .map_err(|e| {
            plan_datafusion_err!(
                "cannot decode an input plan ({e}). Inputs must be serializable \
                 with datafusion-proto: file-sourced DataFrames under standard operators. \
                 In-memory tables and Python UDFs are not supported on DataFusion 55."
            )
        })
}

fn context_for(config: &ConfigOptions) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_config(SessionConfig::from(config.clone()))
        .with_default_features()
        .build();
    SessionContext::new_with_state(state)
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

/// A node: N physical input plans in, one physical plan out.
///
/// Inputs arrive planned (filters and projections pushed into their scans)
/// and pruned to [`PlanNode::required_columns`]; the node refers to their
/// columns by name.
pub trait PlanNode: fmt::Debug + Send + Sync + 'static {
    /// The output schema for these input schemas; validate arguments here.
    fn schema(&self, inputs: &[SchemaRef]) -> Result<SchemaRef>;

    /// Columns each input must keep to produce the output columns
    /// `projection` (indices into [`PlanNode::schema`]); `None` keeps all.
    fn required_columns(
        &self,
        inputs: &[SchemaRef],
        _projection: &[usize],
    ) -> Result<Vec<Option<Vec<String>>>> {
        Ok(vec![None; inputs.len()])
    }

    /// The plan producing the output columns `projection`, in that order.
    /// `schemas` are the full input schemas, `inputs` the pruned plans.
    fn build(
        &self,
        schemas: &[SchemaRef],
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        projection: &[usize],
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>>;
}

/// A [`PlanNode`] over serialized logical plans, as a table provider.
#[derive(Debug, Clone)]
pub struct PlanBytesTable {
    inputs: Vec<Arc<[u8]>>,
    schemas: Vec<SchemaRef>,
    node: Arc<dyn PlanNode>,
    schema: SchemaRef,
    runtime: Option<tokio::runtime::Handle>,
}

impl PlanBytesTable {
    /// Decode the plans (for their schemas) and validate the node.
    pub fn try_new(
        inputs: Vec<Vec<u8>>,
        node: Arc<dyn PlanNode>,
        runtime: Option<tokio::runtime::Handle>,
    ) -> Result<Self> {
        let ctx = context_for(&ConfigOptions::default());
        let schemas = inputs
            .iter()
            .map(|b| Ok(Arc::new(decode_plan(b, &ctx)?.schema().as_arrow().clone())))
            .collect::<Result<Vec<SchemaRef>>>()?;
        let schema = node.schema(&schemas)?;
        Ok(Self {
            inputs: inputs.into_iter().map(Into::into).collect(),
            schemas,
            node,
            schema,
            runtime,
        })
    }

    /// Plan the inputs under `config` and build the node.
    pub async fn plan(
        &self,
        config: &ConfigOptions,
        projection: Option<&Vec<usize>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let all: Vec<usize> = (0..self.schema.fields().len()).collect();
        let projection = projection.map(|p| p.as_slice()).unwrap_or(&all);
        let ctx = context_for(config);
        let state = ctx.state();
        let needed = self.node.required_columns(&self.schemas, projection)?;
        let mut inputs = Vec::with_capacity(self.inputs.len());
        for (bytes, cols) in self.inputs.iter().zip(needed) {
            let mut lp = decode_plan(bytes, &ctx)?;
            if let Some(cols) = cols {
                lp = project(lp, &cols)?;
            }
            inputs.push(state.create_physical_plan(&lp).await?);
        }
        let inner = self.node.build(&self.schemas, inputs, projection, config)?;
        set_last_plan(&inner);
        Ok(Arc::new(BridgeExec::new(inner, self.runtime.clone())))
    }
}

#[async_trait]
impl TableProvider for PlanBytesTable {
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
            // Plan on our own runtime: the caller may be another library
            // (with its own tokio) polling this future on its threads.
            Some(rt) => {
                let this = self.clone();
                rt.spawn(async move { this.plan(&config, projection.as_ref()).await })
                    .await
                    .map_err(|e| datafusion::common::DataFusionError::External(Box::new(e)))?
            }
            None => self.plan(&config, projection.as_ref()).await,
        }
    }
}
