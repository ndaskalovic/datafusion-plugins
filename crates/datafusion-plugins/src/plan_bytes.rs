//! `join_asof` of two serialized logical plans.
//!
//! This is how DataFrames from another DataFusion instance (for example
//! datafusion-python, across its FFI boundary) become join inputs: the
//! caller serializes each input's logical plan with datafusion-proto
//! (`DataFrame.logical_plan().to_bytes()` in Python), and [`AsofPlanTable`]
//! decodes and plans them in its own `SessionContext`, configured from the
//! session that scans it. Planning locally means filter and projection
//! pushdown into the input scans, and their declared orderings, are fully
//! visible to the join.
//!
//! Only plans datafusion-proto can serialize work: file scans (Parquet, CSV,
//! JSON, ...) under any standard operators. In-memory tables and Python UDFs
//! cannot be serialized on DataFusion 55. Files are read from the local
//! filesystem.

use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
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

use crate::plan::{AsofJoinScanExec, JoinAsof, OutputColumn, Side, set_last_plan};

/// Decode a datafusion-proto logical plan in `ctx`.
pub(crate) fn decode_plan(bytes: &[u8], ctx: &SessionContext) -> Result<LogicalPlan> {
    datafusion_proto::bytes::logical_plan_from_bytes(bytes, ctx.task_ctx().as_ref())
        .map_err(|e| {
            plan_datafusion_err!(
                "join_asof: cannot decode an input plan ({e}). Inputs must be serializable \
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
                .ok_or_else(|| plan_datafusion_err!("join_asof: column '{c}' not found"))?;
            Ok(Expr::Column(Column::from(schema.qualified_field(i))))
        })
        .collect::<Result<Vec<_>>>()?;
    LogicalPlanBuilder::from(plan).project(exprs)?.build()
}

/// The table: an ASOF join of two serialized logical plans.
#[derive(Debug, Clone)]
pub struct AsofPlanTable {
    left: Arc<[u8]>,
    right: Arc<[u8]>,
    left_schema: SchemaRef,
    right_schema: SchemaRef,
    builder: JoinAsof,
    outputs: Vec<OutputColumn>,
    schema: SchemaRef,
    runtime: Option<tokio::runtime::Handle>,
}

impl AsofPlanTable {
    /// Decode both plans (for their schemas) and check the join keys.
    pub fn try_new(
        left: Vec<u8>,
        right: Vec<u8>,
        builder: JoinAsof,
        runtime: Option<tokio::runtime::Handle>,
    ) -> Result<Self> {
        let ctx = context_for(&ConfigOptions::default());
        let ls: Schema = decode_plan(&left, &ctx)?.schema().as_arrow().clone();
        let rs: Schema = decode_plan(&right, &ctx)?.schema().as_arrow().clone();
        builder.validate(&ls, &rs)?;
        let outputs = builder.output_columns(&ls, &rs)?;
        let schema = crate::plan::output_schema(&outputs, &ls, &rs)?;
        Ok(Self {
            left: left.into(),
            right: right.into(),
            left_schema: Arc::new(ls),
            right_schema: Arc::new(rs),
            builder,
            outputs,
            schema,
            runtime,
        })
    }

    /// Plan both inputs under `config` and build the join.
    pub async fn plan(
        &self,
        config: &ConfigOptions,
        projection: Option<&Vec<usize>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let outputs: Vec<OutputColumn> = match projection {
            Some(p) => p.iter().map(|i| self.outputs[*i].clone()).collect(),
            None => self.outputs.clone(),
        };
        let ctx = context_for(config);
        let b = &self.builder;
        let lkeys: Vec<&String> = std::iter::once(&b.left_on).chain(&b.by_left).collect();
        let rkeys: Vec<&String> = std::iter::once(&b.right_on).chain(&b.by_right).collect();
        let needed = |schema: &Schema, s: Side, keys: &[&String]| -> Vec<String> {
            schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|n| keys.contains(&n) || outputs.iter().any(|o| o.side == s && &o.column == n))
                .collect()
        };
        let lp = project(decode_plan(&self.left, &ctx)?, &needed(&self.left_schema, Side::Left, &lkeys))?;
        let rp = project(decode_plan(&self.right, &ctx)?, &needed(&self.right_schema, Side::Right, &rkeys))?;
        let state = ctx.state();
        let left = state.create_physical_plan(&lp).await?;
        let right = state.create_physical_plan(&rp).await?;
        let inner = self.builder.build_with_outputs(left, right, &outputs, config)?;
        set_last_plan(&inner);
        Ok(Arc::new(AsofJoinScanExec::new(inner, self.runtime.clone())))
    }
}

#[async_trait]
impl TableProvider for AsofPlanTable {
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
