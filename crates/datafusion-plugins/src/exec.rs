//! `StreamingAsOfJoinExec`: a streaming backward ASOF join (polars
//! `join_asof(strategy="backward")` semantics).
//!
//! Both inputs are sorted by the merge key K = `(by..., ts)`; without `by`
//! every row is in group 0, so K is `ts` and the same kernel runs. One right
//! candidate (the last row passed) is carried per cursor: no hash table.
//!
//! # Range-parallel mode
//!
//! Both inputs are file scans whose partitions are consecutive byte ranges of
//! files sorted by K. DataFusion has no plan property for "partitions are an
//! ordered concatenation", so the operator establishes it at run time:
//!
//! 1. **Prologue** (once per execution, shared by all output partitions):
//!    peek the first batch of every left partition (kept, not re-read) and the
//!    first key of every right partition (in parallel tokio tasks). Right
//!    partitions are sorted by first key; consecutive order is then validated
//!    while streaming (`last(q) <= first(q+1)`).
//! 2. **One worker per left partition** (= output partition), spawned as a
//!    tokio task. It binary-searches the right partition containing the
//!    predecessor of its first left key, then runs a two-cursor merge over
//!    `(left, right)` batches with galloping search on packed i128
//!    `(group, ts)` keys, advancing through the following right partitions
//!    until its left input is exhausted. It also jumps forward over whole
//!    right partitions when the next left key is past the next partition's
//!    first key.
//!
//! # Execute-once mode
//!
//! For inputs that must not be executed twice; see the section below.

use std::any::Any;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use arrow::array::RecordBatch;
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, SchemaRef};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{
    DataFusionError, JoinSide, JoinType, Result, exec_err, internal_datafusion_err, plan_err,
    project_schema,
};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::projection::ProjectionMapping;
use datafusion::physical_expr::{
    LexOrdering, OrderingRequirements, PhysicalExpr, PhysicalExprRef, PhysicalSortExpr,
};
use datafusion::physical_expr_common::physical_expr::fmt_sql;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::joins::utils::{ColumnIndex, build_join_schema};

use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet, Time,
};
use datafusion::physical_plan::projection::{
    EmbeddedProjection, ProjectionExec, try_embed_projection,
};
use datafusion::physical_plan::stream::{RecordBatchReceiverStreamBuilder, RecordBatchStreamAdapter};
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan,
    ExecutionPlanProperties, InputDistributionRequirements, PlanProperties,
    ReplaceChildrenOptions, SendableRecordBatchStream, apply_expression_roots,
};
use futures::StreamExt;
use tokio::sync::mpsc::Sender;

use crate::kernel::{
    NONE, OutCol, RightStore, build_output, gallop_by, left_keys, make_ref, right_keys,
};
use crate::keys::{
    GroupKind, Key, Keys, NoGroup, PrimGroup, RowGroup, TsNative, cmp_keys, is_i32_like,
    packed_group,
    is_primitive_group_type,
};

/// Group key representation chosen from the `by` types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupSpec {
    None,
    I32,
    I64,
    Rows(Vec<DataType>),
}

/// Tunables (`AsofOptions::from_env` reads them from `DFASOF_*` variables).
#[derive(Debug, Clone)]
pub struct AsofOptions {
    /// Right partitions = target_partitions * this (finer split = finer seek).
    pub right_split_factor: usize,
    /// Right batches retained before compaction.
    pub max_sources: usize,
    /// Check the row order inside every left / right batch. Off when the
    /// order is proven (declared and kept by the plan), as polars trusts its
    /// sorted flags. Batch and partition boundaries are always checked
    /// (O(1)): that guards this operator's own assumption that byte-range
    /// partitions of a sorted scan are consecutive key ranges.
    pub validate_left: bool,
    pub validate_right: bool,
    /// Left partitions = target_partitions * this: the work units of the
    /// ordered output's sliding window (finer = better load balance).
    pub left_split_factor: usize,
    /// Emit one ordered partition via `OrderedConcatExec` instead of letting
    /// DataFusion add a SortPreservingMergeExec (which serialises the workers).
    pub ordered_output: bool,
    /// Left partitions in flight in the ordered output = target_partitions + this.
    pub window_extra: usize,
    /// Fan a left partition out over right sub-ranges when fewer than this
    /// many left partitions are non-empty (default: target_partitions).
    pub fanout_below: Option<usize>,
}

impl Default for AsofOptions {
    fn default() -> Self {
        Self {
            right_split_factor: 32,
            max_sources: 64,
            validate_left: false,
            validate_right: false,
            left_split_factor: 4,
            ordered_output: true,
            window_extra: 2,
            fanout_below: None,
        }
    }
}

impl AsofOptions {
    pub fn from_env() -> Self {
        let mut o = Self::default();
        if let Some(v) = std::env::var("DFASOF_RIGHT_SPLIT_FACTOR")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            o.right_split_factor = v;
        }
        if let Some(v) = std::env::var("DFASOF_MAX_SOURCES")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            o.max_sources = v;
        }
        if let Some(v) = std::env::var("DFASOF_LEFT_SPLIT_FACTOR")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            o.left_split_factor = v;
        }
        if let Some(v) = std::env::var("DFASOF_WINDOW_EXTRA")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            o.window_extra = v;
        }
        if let Some(v) = std::env::var("DFASOF_FANOUT_BELOW")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            o.fanout_below = Some(v);
        }
        if let Ok(v) = std::env::var("DFASOF_ORDERED") {
            o.ordered_output = !(v == "0" || v.eq_ignore_ascii_case("false"));
        }
        if let Ok(v) = std::env::var("DFASOF_VALIDATE") {
            let on = v == "1" || v.eq_ignore_ascii_case("true");
            o.validate_left |= on;
            o.validate_right |= on;
        }
        o
    }
}

/// Everything a worker needs to know about the join.
#[derive(Debug)]
pub struct JoinSpec {
    pub left_ts: PhysicalExprRef,
    pub right_ts: PhysicalExprRef,
    pub left_by: Vec<PhysicalExprRef>,
    pub right_by: Vec<PhysicalExprRef>,
    /// `>` instead of `>=`.
    pub strict: bool,
    /// `(by..., ts)` of each side, for error messages.
    pub left_key_desc: String,
    pub right_key_desc: String,
    pub out_schema: SchemaRef,
    pub out_cols: Vec<OutCol>,
    /// Right columns kept for output (the "payload").
    pub payload_idx: Vec<usize>,
    pub payload_schema: SchemaRef,
    pub options: AsofOptions,
}

/// The operator.
pub struct StreamingAsOfJoinExec {
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
    left_ts: PhysicalExprRef,
    right_ts: PhysicalExprRef,
    strict: bool,
    /// Execute-once mode (see [`StreamingAsOfJoinExec::with_execute_once`]).
    once: bool,
    /// Execute-once: right partitions are consecutive key ranges (one unit
    /// each); otherwise they are read in order as one unit.
    right_units: bool,
    left_nulls_first: bool,
    right_nulls_first: bool,
    left_ordering: LexOrdering,
    right_ordering: LexOrdering,
    join_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    spec: Arc<JoinSpec>,
    group_spec: GroupSpec,
    ts_i32: bool,
    options: AsofOptions,
    metrics: ExecutionPlanMetricsSet,
    /// Per-execution shared state (`SharedRun<TN, G>`), created by the first
    /// `execute` call.
    shared: Mutex<Option<Arc<dyn Any + Send + Sync>>>,
    cache: Arc<PlanProperties>,
}

impl fmt::Debug for StreamingAsOfJoinExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamingAsOfJoinExec")
            .field("strict", &self.strict)
            .field("once", &self.once)
            .field("projection", &self.projection)
            .finish()
    }
}

fn key_desc(exprs: &[PhysicalExprRef]) -> String {
    let names: Vec<String> = exprs.iter().map(|e| fmt_sql(e.as_ref()).to_string()).collect();
    format!("({})", names.join(", "))
}

fn sort_exprs(exprs: &[PhysicalExprRef], nulls_first: bool) -> Vec<PhysicalSortExpr> {
    exprs
        .iter()
        .map(|e| PhysicalSortExpr {
            expr: Arc::clone(e),
            options: SortOptions {
                descending: false,
                nulls_first,
            },
        })
        .collect()
}

impl StreamingAsOfJoinExec {
    /// Build the operator. `on` pairs are `(left, right)` equality keys;
    /// the match condition is `left_ts >= right_ts` (or `>` when `strict`).
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
        left_ts: PhysicalExprRef,
        right_ts: PhysicalExprRef,
        strict: bool,
        left_nulls_first: bool,
        right_nulls_first: bool,
        projection: Option<Vec<usize>>,
        options: AsofOptions,
    ) -> Result<Self> {
        let left_schema = left.schema();
        let right_schema = right.schema();
        let ts_type = left_ts.data_type(&left_schema)?;
        if right_ts.data_type(&right_schema)? != ts_type {
            return plan_err!("StreamingAsOfJoin: match expressions have different types");
        }
        let mut by_types = Vec::with_capacity(on.len());
        for (l, r) in &on {
            let lt = l.data_type(&left_schema)?;
            if r.data_type(&right_schema)? != lt {
                return plan_err!("StreamingAsOfJoin: equality keys have different types");
            }
            by_types.push(lt);
        }
        let group_spec = match by_types.as_slice() {
            [] => GroupSpec::None,
            [t] if is_i32_like(t) => GroupSpec::I32,
            [t] if is_primitive_group_type(t) => GroupSpec::I64,
            ts => {
                if !RowGroup::supports(ts) {
                    return plan_err!("StreamingAsOfJoin: unsupported equality key types");
                }
                GroupSpec::Rows(ts.to_vec())
            }
        };
        let ts_i32 = is_i32_like(&ts_type);

        let (left_by, right_by): (Vec<_>, Vec<_>) = on.iter().cloned().unzip();
        let mut lk: Vec<PhysicalExprRef> = left_by.clone();
        let mut rk: Vec<PhysicalExprRef> = right_by.clone();
        lk.push(Arc::clone(&left_ts));
        rk.push(Arc::clone(&right_ts));
        let left_ordering = LexOrdering::new(sort_exprs(&lk, left_nulls_first))
            .ok_or_else(|| internal_datafusion_err!("empty left ordering"))?;
        let right_ordering = LexOrdering::new(sort_exprs(&rk, right_nulls_first))
            .ok_or_else(|| internal_datafusion_err!("empty right ordering"))?;

        let (join_schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, &JoinType::Left);
        let join_schema = Arc::new(join_schema);
        let out_schema = project_schema(&join_schema, projection.as_ref())?;
        let projected: Vec<ColumnIndex> = match &projection {
            Some(p) => p.iter().map(|i| column_indices[*i].clone()).collect(),
            None => column_indices.clone(),
        };
        let mut payload_idx: Vec<usize> = Vec::new();
        let mut out_cols = Vec::with_capacity(projected.len());
        for c in &projected {
            match c.side {
                JoinSide::Left => out_cols.push(OutCol::Left(c.index)),
                JoinSide::Right => {
                    let j = match payload_idx.iter().position(|x| *x == c.index) {
                        Some(j) => j,
                        None => {
                            payload_idx.push(c.index);
                            payload_idx.len() - 1
                        }
                    };
                    out_cols.push(OutCol::Right(j));
                }
                JoinSide::None => {
                    return plan_err!("StreamingAsOfJoin: unexpected mark column");
                }
            }
        }
        let payload_schema = Arc::new(right_schema.project(&payload_idx)?);
        let spec = Arc::new(JoinSpec {
            left_ts: Arc::clone(&left_ts),
            right_ts: Arc::clone(&right_ts),
            left_by,
            right_by,
            strict,
            left_key_desc: key_desc(&lk),
            right_key_desc: key_desc(&rk),
            out_schema,
            out_cols,
            payload_idx,
            payload_schema,
            options: options.clone(),
        });
        let cache = Arc::new(Self::compute_properties(
            &left,
            &join_schema,
            projection.as_deref(),
        )?);
        Ok(Self {
            left,
            right,
            on,
            left_ts,
            right_ts,
            strict,
            once: false,
            right_units: false,
            left_nulls_first,
            right_nulls_first,
            left_ordering,
            right_ordering,
            join_schema,
            projection,
            spec,
            group_spec,
            ts_i32,
            options,
            metrics: ExecutionPlanMetricsSet::new(),
            shared: Mutex::new(None),
            cache,
        })
    }

    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    fn rebuild(
        &self,
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let (once, right_units) = (self.once, self.right_units);
        let new = Self::try_new(
            left,
            right,
            self.on.clone(),
            Arc::clone(&self.left_ts),
            Arc::clone(&self.right_ts),
            self.strict,
            self.left_nulls_first,
            self.right_nulls_first,
            projection,
            self.options.clone(),
        )?;
        Ok(if once { new.with_execute_once(right_units) } else { new })
    }

    /// Switch to execute-once mode: every partition of both inputs is
    /// executed exactly once, so any upstream plan (joins, aggregates,
    /// repartitions) is safe. Left partitions are read in order as one
    /// stream. If `right_units`, right partitions are consecutive, disjoint
    /// key ranges and each becomes a unit worked by one task (a serial
    /// distributor slices the left stream by unit boundaries); otherwise they
    /// are read in order as one unit. The output is one ordered partition.
    pub fn with_execute_once(mut self, right_units: bool) -> Self {
        self.once = true;
        self.right_units = right_units;
        let p = Arc::make_mut(&mut self.cache);
        p.partitioning = datafusion::physical_plan::Partitioning::UnknownPartitioning(1);
        self
    }

    pub fn is_execute_once(&self) -> bool {
        self.once
    }

    /// Output = left's order, partitioning and equivalences, projected into
    /// the join schema (same as DataFusion's `AsOfJoinExec`).
    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        join_schema: &SchemaRef,
        projection: Option<&[usize]>,
    ) -> Result<PlanProperties> {
        let left_schema = left.schema();
        let mapping = ProjectionMapping::try_new(
            left_schema
                .fields()
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    (
                        Arc::new(Column::new(field.name(), index)) as PhysicalExprRef,
                        field.name().to_string(),
                    )
                }),
            &left_schema,
        )?;
        let input_eq_properties = left.equivalence_properties();
        let mut eq_properties = input_eq_properties.project(&mapping, Arc::clone(join_schema));
        let mut output_partitioning = left
            .output_partitioning()
            .project(&mapping, input_eq_properties);
        if let Some(projection) = projection {
            let projection_mapping = ProjectionMapping::from_indices(projection, join_schema)?;
            let output_schema = project_schema(join_schema, Some(&projection))?;
            output_partitioning = output_partitioning.project(&projection_mapping, &eq_properties);
            eq_properties = eq_properties.project(&projection_mapping, output_schema);
        }
        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }

    /// Get (or create) the shared state of the current execution.
    fn shared_run<TN: TsNative, G: GroupKind>(
        &self,
        partition: usize,
    ) -> Result<Arc<SharedRun<TN, G>>> {
        let n_left = self.left.output_partitioning().partition_count();
        let mut guard = self
            .shared
            .lock()
            .map_err(|_| internal_datafusion_err!("StreamingAsOfJoin: poisoned lock"))?;
        if let Some(any) = guard.as_ref()
            && let Ok(run) = Arc::clone(any).downcast::<SharedRun<TN, G>>()
            && run.claimed.len() == n_left
            && !run.claimed[partition].swap(true, AtomicOrdering::SeqCst)
        {
            return Ok(run);
        }
        // First call, or a re-execution of a partition already started: begin
        // a fresh execution.
        let run = Arc::new(SharedRun::<TN, G>::new(n_left));
        run.claimed[partition].store(true, AtomicOrdering::SeqCst);
        *guard = Some(Arc::clone(&run) as Arc<dyn Any + Send + Sync>);
        Ok(run)
    }

    fn start<TN: TsNative, G: GroupKind>(
        &self,
        group: G,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if self.once {
            return self.start_once::<TN, G>(group, partition, context);
        }
        let run = self.shared_run::<TN, G>(partition)?;
        let mut builder = RecordBatchReceiverStreamBuilder::new(self.schema(), 2);
        let tx = builder.tx();
        let metrics = BaselineMetrics::new(&self.metrics, partition);
        let reservation = MemoryConsumer::new(format!("StreamingAsOfJoin[{partition}]"))
            .register(context.memory_pool());
        let args = WorkerArgs {
            p: partition,
            run,
            spec: Arc::clone(&self.spec),
            group: Arc::new(group),
            left: Arc::clone(&self.left),
            right: Arc::clone(&self.right),
            ctx: context,
            tx,
            metrics,
            reservation,
        };
        builder.spawn(async move { run_worker::<TN, G>(args).await });
        Ok(builder.build())
    }

    fn start_once<TN: TsNative, G: GroupKind>(
        &self,
        group: G,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return exec_err!("StreamingAsOfJoin (execute-once) has one partition, got {partition}");
        }
        let mut builder = RecordBatchReceiverStreamBuilder::new(self.schema(), 2);
        let args = OnceArgs {
            spec: Arc::clone(&self.spec),
            group: Arc::new(group),
            left: Arc::clone(&self.left),
            right: Arc::clone(&self.right),
            right_units: self.right_units,
            ctx: context,
            tx: builder.tx(),
            metrics: BaselineMetrics::new(&self.metrics, partition),
        };
        builder.spawn(async move { run_once::<TN, G>(args).await });
        Ok(builder.build())
    }
}

impl EmbeddedProjection for StreamingAsOfJoinExec {
    fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        self.rebuild(Arc::clone(&self.left), Arc::clone(&self.right), projection)
    }
}

impl DisplayAs for StreamingAsOfJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let on = self
            .on
            .iter()
            .map(|(l, r)| format!("({} = {})", fmt_sql(l.as_ref()), fmt_sql(r.as_ref())))
            .collect::<Vec<_>>()
            .join(", ");
        let op = if self.strict { ">" } else { ">=" };
        let m = format!(
            "{} {op} {}",
            fmt_sql(self.left_ts.as_ref()),
            fmt_sql(self.right_ts.as_ref())
        );
        let projection = self
            .projection
            .as_ref()
            .map(|p| {
                format!(
                    ", projection=[{}]",
                    p.iter()
                        .map(|i| format!("{}@{}", self.join_schema.field(*i).name(), i))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .unwrap_or_default();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "StreamingAsOfJoinExec: {}on=[{on}], match=[{m}]{projection}",
                match (self.once, self.right_units) {
                    (false, _) => "",
                    (true, true) => "execute_once (right units), ",
                    (true, false) => "execute_once, ",
                },
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "on={on}")?;
                writeln!(f, "match={m}")
            }
        }
    }
}

impl ExecutionPlan for StreamingAsOfJoinExec {
    fn name(&self) -> &str {
        "StreamingAsOfJoinExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    #[allow(deprecated)]
    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.input_distribution_requirements().into_per_child()
    }

    fn input_distribution_requirements(&self) -> InputDistributionRequirements {
        // No particular distribution: each worker seeks into the right input
        // itself, and partitions only have to be ordered internally.
        let d = Distribution::UnspecifiedDistribution;
        InputDistributionRequirements::new(vec![d.clone(), d])
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![
            Some(OrderingRequirements::from(self.left_ordering.clone())),
            Some(OrderingRequirements::from(self.right_ordering.clone())),
        ]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true, false]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // Never let EnforceDistribution add a round-robin repartition: it
        // would destroy the order. File scans are still split into byte
        // ranges (that happens regardless of this flag).
        vec![false, false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let keys = self.on.iter().flat_map(|(l, r)| [l, r]);
        apply_expression_roots(keys.chain([&self.left_ts, &self.right_ts]), f)
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return plan_err!("StreamingAsOfJoinExec expects two children");
        }
        let right = children.swap_remove(1);
        let left = children.swap_remove(0);
        match options.children_properties {
            ChildrenPropertiesMode::Keep => {
                let mut new = self.rebuild(left, right, self.projection.clone())?;
                new.cache = Arc::clone(&self.cache);
                Ok(Arc::new(new))
            }
            ChildrenPropertiesMode::Recompute => {
                Ok(Arc::new(self.rebuild(left, right, self.projection.clone())?))
            }
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        match (self.ts_i32, &self.group_spec) {
            (false, GroupSpec::None) => self.start::<i64, NoGroup>(NoGroup, partition, context),
            (false, GroupSpec::I32) => {
                self.start::<i64, PrimGroup<i32>>(PrimGroup::new(), partition, context)
            }
            (false, GroupSpec::I64) => {
                self.start::<i64, PrimGroup<i64>>(PrimGroup::new(), partition, context)
            }
            (false, GroupSpec::Rows(t)) => {
                self.start::<i64, RowGroup>(RowGroup::try_new(t)?, partition, context)
            }
            (true, GroupSpec::None) => self.start::<i32, NoGroup>(NoGroup, partition, context),
            (true, GroupSpec::I32) => {
                self.start::<i32, PrimGroup<i32>>(PrimGroup::new(), partition, context)
            }
            (true, GroupSpec::I64) => {
                self.start::<i32, PrimGroup<i64>>(PrimGroup::new(), partition, context)
            }
            (true, GroupSpec::Rows(t)) => {
                self.start::<i32, RowGroup>(RowGroup::try_new(t)?, partition, context)
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if self.projection.is_some() {
            return Ok(None);
        }
        try_embed_projection(projection, self)
    }
}

// ---------------------------------------------------------------------------
// Shared per-execution state
// ---------------------------------------------------------------------------

pub(crate) type LeftStart = (VecDeque<RecordBatch>, Option<SendableRecordBatchStream>);

pub(crate) struct Prologue<TN: TsNative, G: GroupKind> {
    /// Peeked batches + remaining stream of each left partition (taken once).
    pub(crate) left: Mutex<Vec<Option<LeftStart>>>,
    /// First valid key of each left partition (`None` = no valid rows).
    pub(crate) left_first: Vec<Option<Key<TN, G>>>,
    /// Non-empty right partitions sorted by first key...
    pub(crate) right_order: Vec<usize>,
    /// ...and their first keys (same order).
    pub(crate) right_first: Vec<Key<TN, G>>,
}

pub(crate) struct SharedRun<TN: TsNative, G: GroupKind> {
    pub(crate) prologue: tokio::sync::OnceCell<Arc<Prologue<TN, G>>>,
    pub(crate) claimed: Vec<AtomicBool>,
}

impl<TN: TsNative, G: GroupKind> SharedRun<TN, G> {
    pub(crate) fn new(n_left: usize) -> Self {
        Self {
            prologue: tokio::sync::OnceCell::new(),
            claimed: (0..n_left).map(|_| AtomicBool::new(false)).collect(),
        }
    }
}

pub(crate) fn join_err(e: tokio::task::JoinError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Read the first key of right partition `q` (first valid row of its first
/// non-empty batch). The stream is dropped afterwards.
async fn probe_right<TN: TsNative, G: GroupKind>(
    right: Arc<dyn ExecutionPlan>,
    q: usize,
    ctx: Arc<TaskContext>,
    spec: Arc<JoinSpec>,
    g: Arc<G>,
) -> Result<Option<Key<TN, G>>> {
    let mut s = right.execute(q, ctx)?;
    while let Some(b) = s.next().await {
        let b = b?;
        if let Some((k, _)) =
            right_keys::<TN, G>(&*g, &spec.right_ts, &spec.right_by, &[], &b)?
        {
            return Ok(Some(k.key_at(&*g, 0)));
        }
    }
    Ok(None)
}

/// Peek a left partition until its first valid key.
async fn peek_left<TN: TsNative, G: GroupKind>(
    left: Arc<dyn ExecutionPlan>,
    p: usize,
    ctx: Arc<TaskContext>,
    spec: Arc<JoinSpec>,
    g: Arc<G>,
) -> Result<(LeftStart, Option<Key<TN, G>>)> {
    let mut s = left.execute(p, ctx)?;
    let mut batches = VecDeque::new();
    while let Some(b) = s.next().await {
        let b = b?;
        if b.num_rows() == 0 {
            continue;
        }
        let k = left_keys::<TN, G>(&*g, &spec.left_ts, &spec.left_by, &b)?;
        let first = k.first_valid().map(|i| k.key_at(&*g, i));
        batches.push_back(b);
        if let Some(first) = first {
            return Ok(((batches, Some(s)), Some(first)));
        }
    }
    Ok(((batches, None), None))
}

pub(crate) async fn init_prologue<TN: TsNative, G: GroupKind>(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
    spec: Arc<JoinSpec>,
    g: Arc<G>,
) -> Result<Arc<Prologue<TN, G>>> {
    let n_left = left.output_partitioning().partition_count();
    let n_right = right.output_partitioning().partition_count();
    let trace = std::env::var_os("DFASOF_TRACE").is_some();
    let t0 = std::time::Instant::now();

    // Left peeks and right probes all run as parallel tasks (each decodes one
    // batch); JoinSet aborts them if this future is dropped.
    let mut lset = tokio::task::JoinSet::new();
    for p in 0..n_left {
        let (l, c, s, gg) = (Arc::clone(&left), Arc::clone(&ctx), Arc::clone(&spec), Arc::clone(&g));
        lset.spawn(async move { (p, peek_left::<TN, G>(l, p, c, s, gg).await) });
    }
    let mut left_starts: Vec<Option<LeftStart>> = (0..n_left).map(|_| None).collect();
    let mut left_first: Vec<Option<Key<TN, G>>> = vec![None; n_left];
    while let Some(res) = lset.join_next().await {
        let (p, r) = res.map_err(join_err)?;
        let (start, first) = r?;
        left_starts[p] = Some(start);
        left_first[p] = first;
    }

    if trace {
        eprintln!("[dfasof] left peeks: {n_left} partitions in {:?}", t0.elapsed());
    }
    let t1 = std::time::Instant::now();
    // Right probes, with bounded concurrency: each open probe holds one row
    // group's column chunks in memory.
    let mut right_keys_by_part: Vec<Option<Key<TN, G>>> = vec![None; n_right];
    if n_right > 1 {
        let threads = ctx.session_config().target_partitions().max(1);
        let limit = (threads * 2).min(n_right);
        type ProbeOut<TN, G> = (usize, Result<Option<Key<TN, G>>>);
        let mut rset: tokio::task::JoinSet<ProbeOut<TN, G>> = tokio::task::JoinSet::new();
        let spawn_probe = |set: &mut tokio::task::JoinSet<ProbeOut<TN, G>>, q: usize| {
            let (r, c, s, gg) =
                (Arc::clone(&right), Arc::clone(&ctx), Arc::clone(&spec), Arc::clone(&g));
            set.spawn(async move { (q, probe_right::<TN, G>(r, q, c, s, gg).await) });
        };
        let mut next_q = 0;
        while next_q < limit {
            spawn_probe(&mut rset, next_q);
            next_q += 1;
        }
        while let Some(res) = rset.join_next().await {
            let (q, r) = res.map_err(join_err)?;
            right_keys_by_part[q] = r?;
            if next_q < n_right {
                spawn_probe(&mut rset, next_q);
                next_q += 1;
            }
        }
    }
    let mut order: Vec<(Key<TN, G>, usize)> = if n_right > 1 {
        right_keys_by_part
            .into_iter()
            .enumerate()
            .filter_map(|(q, k)| k.map(|k| (k, q)))
            .collect()
    } else {
        // Single right partition: no probe needed; its "first key" is never
        // compared (binary search over one element returns it).
        Vec::new()
    };
    if trace {
        eprintln!("[dfasof] right probes: {n_right} partitions in {:?}", t1.elapsed());
    }
    // Sort by first key; ties keep partition (= file) order.
    order.sort_by(|a, b| cmp_keys::<TN, G>(&a.0, &b.0).then(a.1.cmp(&b.1)));
    let (right_first, right_order): (Vec<_>, Vec<_>) = order.into_iter().unzip();

    Ok(Arc::new(Prologue {
        left: Mutex::new(left_starts),
        left_first,
        right_order,
        right_first,
    }))
}

// ---------------------------------------------------------------------------
// Workers
// ---------------------------------------------------------------------------

struct WorkerArgs<TN: TsNative, G: GroupKind> {
    p: usize,
    run: Arc<SharedRun<TN, G>>,
    spec: Arc<JoinSpec>,
    group: Arc<G>,
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
    tx: Sender<Result<RecordBatch>>,
    metrics: BaselineMetrics,
    reservation: MemoryReservation,
}

struct Worker<TN: TsNative, G: GroupKind> {
    p: usize,
    pro: Arc<Prologue<TN, G>>,
    spec: Arc<JoinSpec>,
    group: Arc<G>,
    right_plan: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
    tx: Sender<Result<RecordBatch>>,
    metrics: BaselineMetrics,
    elapsed: Time,
    reservation: MemoryReservation,
    validate_left: bool,
    validate_right: bool,

    // Right cursor.
    /// Index into `pro.right_order` of the open partition.
    cur_idx: usize,
    stream: Option<SendableRecordBatchStream>,
    rkeys: Option<Keys<TN, G>>,
    rpos: usize,
    rsrc: u32,
    right_eof: bool,
    right_last: Option<Key<TN, G>>,
    store: RightStore,

    // The candidate: the last right row passed.
    cand: u64,
    cand_group: Option<G::Owned>,
    /// Group half of the candidate's packed key (packed keys only).
    cand_gpk: i64,

    left_last: Option<Key<TN, G>>,
    /// Sub-worker of the sparse-left fan-out: its first valid left key
    /// (instead of the partition's).
    first_override: Option<Option<Key<TN, G>>>,

    // Execute-once mode.
    /// The right input is this worker's unit only (never re-executed).
    once: bool,
    /// The unit's peeked batches and remaining stream (taken by `open`).
    unit: Option<LeftStart>,
    /// First key of the next unit: the unit's last key must not exceed it.
    next_first: Option<Key<TN, G>>,
}

async fn run_worker<TN: TsNative, G: GroupKind>(a: WorkerArgs<TN, G>) -> Result<()> {
    let pro = {
        let (l, r, c, s, g) = (
            Arc::clone(&a.left),
            Arc::clone(&a.right),
            Arc::clone(&a.ctx),
            Arc::clone(&a.spec),
            Arc::clone(&a.group),
        );
        let run = Arc::clone(&a.run);
        Arc::clone(
            run.prologue
                .get_or_try_init(|| init_prologue::<TN, G>(l, r, c, s, g))
                .await?,
        )
    };
    let start = {
        let mut guard = pro
            .left
            .lock()
            .map_err(|_| internal_datafusion_err!("StreamingAsOfJoin: poisoned lock"))?;
        guard[a.p].take()
    };
    let Some((peeked, stream)) = start else {
        return exec_err!("StreamingAsOfJoin: partition {} executed twice", a.p);
    };
    // Fewer non-empty left partitions than threads (a sparse or small left
    // side: its file has few row groups): split this partition's right span
    // across sub-workers instead, or most threads would sit idle.
    let threads = a.ctx.session_config().target_partitions().max(1);
    if pro.right_order.len() > 1
        && pro.left_first.iter().flatten().count() < a.spec.options.fanout_below.unwrap_or(threads)
    {
        return run_fanout(a, pro, peeked, stream, threads).await;
    }
    let mut w = new_worker(
        a.p,
        pro,
        a.spec,
        a.group,
        a.right,
        a.ctx,
        a.tx,
        a.metrics,
        a.reservation,
        None,
    );
    let res = w.run_single(peeked, stream).await;
    w.metrics.done();
    w.reservation.free();
    res
}

#[allow(clippy::too_many_arguments)]
fn new_worker<TN: TsNative, G: GroupKind>(
    p: usize,
    pro: Arc<Prologue<TN, G>>,
    spec: Arc<JoinSpec>,
    group: Arc<G>,
    right: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
    tx: Sender<Result<RecordBatch>>,
    metrics: BaselineMetrics,
    reservation: MemoryReservation,
    first_override: Option<Option<Key<TN, G>>>,
) -> Worker<TN, G> {
    let elapsed = metrics.elapsed_compute().clone();
    Worker::<TN, G> {
        p,
        pro,
        validate_left: spec.options.validate_left,
        validate_right: spec.options.validate_right,
        store: RightStore::new(Arc::clone(&spec.payload_schema), spec.options.max_sources.max(2)),
        spec,
        group,
        right_plan: right,
        ctx,
        tx,
        metrics,
        elapsed,
        reservation,
        cur_idx: 0,
        stream: None,
        rkeys: None,
        rpos: 0,
        rsrc: 0,
        right_eof: false,
        right_last: None,
        cand: NONE,
        cand_group: None,
        cand_gpk: 0,
        left_last: None,
        first_override,
        once: false,
        unit: None,
        next_first: None,
    }
}

/// Sparse-left path: read this left partition fully (it is small), split the
/// right partitions it spans into up to `threads` consecutive sub-ranges, and
/// give each sub-range the left rows whose predecessor lies in it. Sub-workers
/// run in parallel; their outputs are forwarded in order.
///
/// A left row with key k belongs to the last sub-range whose first key is
/// <= k (< k when strict): the same rule as `find_start`, so its predecessor
/// is inside that sub-range. NULL-key rows go with the next valid row (with
/// the last sub-range if none follows), which keeps the left order intact.
async fn run_fanout<TN: TsNative, G: GroupKind>(
    a: WorkerArgs<TN, G>,
    pro: Arc<Prologue<TN, G>>,
    mut peeked: VecDeque<RecordBatch>,
    mut stream: Option<SendableRecordBatchStream>,
    threads: usize,
) -> Result<()> {
    let g = Arc::clone(&a.group);
    let spec = Arc::clone(&a.spec);
    let strict = spec.strict;
    let mut batches = Vec::new();
    let mut keys = Vec::new();
    while let Some(b) = next_left(&mut peeked, &mut stream).await? {
        if b.num_rows() == 0 {
            continue;
        }
        keys.push(left_keys::<TN, G>(&*g, &spec.left_ts, &spec.left_by, &b)?);
        batches.push(b);
    }
    // Valid rows in order: (batch, row).
    let mut valid: Vec<(usize, usize)> = Vec::new();
    for (bi, k) in keys.iter().enumerate() {
        valid.extend((0..k.len).filter(|&r| k.is_valid(r)).map(|r| (bi, r)));
    }
    let key_of = |(bi, r): (usize, usize)| keys[bi].key_at(&*g, r);
    let start_of = |k: &Key<TN, G>| -> usize {
        pro.right_first
            .partition_point(|fk| {
                let o = cmp_keys::<TN, G>(fk, k);
                if strict { o == Ordering::Less } else { o != Ordering::Greater }
            })
            .saturating_sub(1)
    };
    // Cut points as indices into `valid`.
    let mut cuts = vec![0usize];
    if let (Some(&f), Some(&l)) = (valid.first(), valid.last()) {
        let (ia, ib) = (start_of(&key_of(f)), start_of(&key_of(l)));
        let span = ib - ia + 1;
        let s = span.min(threads).max(1);
        for j in 1..s {
            let bound = &pro.right_first[ia + j * span / s];
            // First valid row that belongs to sub-range j.
            let c = valid.partition_point(|&v| {
                let o = cmp_keys::<TN, G>(bound, &key_of(v));
                if strict { o != Ordering::Less } else { o == Ordering::Greater }
            });
            if c > *cuts.last().unwrap() && c < valid.len() {
                cuts.push(c);
            }
        }
    }
    // Global row position of each cut: the cut's valid row, moved back over
    // the NULL rows just before it (they go with the next valid row).
    let mut offsets = vec![0usize; batches.len() + 1];
    for (i, b) in batches.iter().enumerate() {
        offsets[i + 1] = offsets[i] + b.num_rows();
    }
    let total = offsets[batches.len()];
    let mut pos: Vec<usize> = cuts
        .iter()
        .skip(1)
        .map(|&c| {
            let (bi, r) = valid[c];
            let mut gp = offsets[bi] + r;
            let prev_valid = valid[c - 1];
            let prev_gp = offsets[prev_valid.0] + prev_valid.1;
            gp = gp.min(prev_gp + 1).max(prev_gp + 1);
            gp
        })
        .collect();
    pos.insert(0, 0);
    pos.push(total);
    // Slice the batches into one VecDeque per sub-range.
    let slice = |lo: usize, hi: usize| -> VecDeque<RecordBatch> {
        let mut out = VecDeque::new();
        for (i, b) in batches.iter().enumerate() {
            let (s, e) = (offsets[i].max(lo), offsets[i + 1].min(hi));
            if s < e {
                out.push_back(b.slice(s - offsets[i], e - s));
            }
        }
        out
    };
    let mut handles = Vec::new();
    for j in 0..pos.len() - 1 {
        let part = slice(pos[j], pos[j + 1]);
        let first = if j < cuts.len() { valid.get(cuts[j]).map(|&v| key_of(v)) } else { None };
        let (tx, rx) = tokio::sync::mpsc::channel(1024);
        let reservation = MemoryConsumer::new(format!("StreamingAsOfJoin[{}.{j}]", a.p))
            .register(a.ctx.memory_pool());
        let mut w = new_worker(
            a.p,
            Arc::clone(&pro),
            Arc::clone(&spec),
            Arc::clone(&g),
            Arc::clone(&a.right),
            Arc::clone(&a.ctx),
            tx,
            a.metrics.clone(),
            reservation,
            Some(first),
        );
        let h = tokio::spawn(async move {
            let r = w.run_single(part, None).await;
            w.reservation.free();
            r
        });
        handles.push((h, rx));
    }
    drop(batches);
    for (h, mut rx) in handles {
        while let Some(b) = rx.recv().await {
            if a.tx.send(b).await.is_err() {
                return Ok(());
            }
        }
        h.await.map_err(join_err)??;
    }
    a.metrics.done();
    Ok(())
}

pub(crate) async fn next_left(
    peeked: &mut VecDeque<RecordBatch>,
    stream: &mut Option<SendableRecordBatchStream>,
) -> Result<Option<RecordBatch>> {
    if let Some(b) = peeked.pop_front() {
        return Ok(Some(b));
    }
    let Some(s) = stream.as_mut() else {
        return Ok(None);
    };
    match s.next().await {
        Some(b) => Ok(Some(b?)),
        None => {
            *stream = None;
            Ok(None)
        }
    }
}

impl<TN: TsNative, G: GroupKind> Worker<TN, G> {
    // ----- right cursor -----------------------------------------------------

    /// Index (into `right_order`) of the last right partition whose first key
    /// is <= `target` (< when `strict`): the partition holding `target`'s
    /// predecessor. With `>` a partition that starts on a tie with `target`
    /// holds no eligible row, so the predecessor is in an earlier one.
    fn find_start(&self, target: &Key<TN, G>, strict: bool) -> usize {
        let k = self.pro.right_first.partition_point(|fk| {
            let o = cmp_keys::<TN, G>(fk, target);
            if strict { o == Ordering::Less } else { o != Ordering::Greater }
        });
        k.saturating_sub(1)
    }

    fn open(&mut self, idx: usize) -> Result<()> {
        self.stream = None;
        self.rkeys = None;
        if self.once {
            // Execute-once: the unit's stream is the only right input, and it
            // is consumed at most once.
            if idx == 0
                && let Some((peeked, rest)) = self.unit.take()
            {
                self.stream = Some(chain_stream(self.right_plan.schema(), peeked, rest));
            }
            self.cur_idx = idx;
            return Ok(());
        }
        if self.pro.right_order.is_empty() {
            // One right partition (never probed), or all empty.
            let n = self.right_plan.output_partitioning().partition_count();
            if n == 1 && idx == 0 {
                self.stream = Some(self.right_plan.execute(0, Arc::clone(&self.ctx))?);
            }
            self.cur_idx = idx;
            return Ok(());
        }
        if idx < self.pro.right_order.len() {
            let q = self.pro.right_order[idx];
            self.stream = Some(self.right_plan.execute(q, Arc::clone(&self.ctx))?);
        }
        self.cur_idx = idx;
        Ok(())
    }

    /// Single-candidate modes: if `target` lies at or past the first key of a
    /// later right partition, skip straight to the last such partition. Rows
    /// in between can't be the predecessor of `target` or of any later left
    /// key, so they are never decoded.
    fn maybe_jump(&mut self, target: &Key<TN, G>) -> Result<()> {
        if self.pro.right_order.len() <= 1 || self.cur_idx + 1 >= self.pro.right_order.len() {
            return Ok(());
        }
        let next_first = &self.pro.right_first[self.cur_idx + 1];
        let o = cmp_keys::<TN, G>(next_first, target);
        if o == Ordering::Greater || (self.spec.strict && o == Ordering::Equal) {
            return Ok(());
        }
        let want = self.find_start(target, self.spec.strict);
        if want > self.cur_idx {
            self.open(want)?;
        }
        Ok(())
    }

    fn check_right_order(&mut self, k: &Keys<TN, G>) -> Result<()> {
        let first = k.key_at(&*self.group, 0);
        if let Some(last) = &self.right_last
            && cmp_keys::<TN, G>(last, &first) == Ordering::Greater
        {
            return exec_err!(
                "right input is not sorted by {} (a batch or partition starts before the \
                 previous one ended)",
                self.spec.right_key_desc
            );
        }
        if self.validate_right && !k.is_sorted(&self.group) {
            return exec_err!("right input is not sorted by {}", self.spec.right_key_desc);
        }
        self.right_last = Some(k.key_at(&*self.group, k.len - 1));
        Ok(())
    }

    fn check_left_order(&mut self, lk: &Keys<TN, G>) -> Result<()> {
        let Some(f) = lk.first_valid() else {
            return Ok(());
        };
        let l = lk.last_valid().unwrap_or(f);
        let fk = lk.key_at(&*self.group, f);
        if let Some(prev) = &self.left_last
            && cmp_keys::<TN, G>(prev, &fk) == Ordering::Greater
        {
            return exec_err!(
                "left input is not sorted by {} (a batch starts before the previous one ended)",
                self.spec.left_key_desc
            );
        }
        if self.validate_left && !lk.is_sorted(&self.group) {
            return exec_err!("left input is not sorted by {}", self.spec.left_key_desc);
        }
        self.left_last = Some(lk.key_at(&*self.group, l));
        Ok(())
    }

    fn compact(&mut self, pending: &mut [u64]) -> Result<()> {
        let mut cand = [self.cand];
        self.store.compact(&mut [&mut cand[..], pending])?;
        self.cand = cand[0];
        Ok(())
    }

    fn update_reservation(&mut self) -> Result<()> {
        self.reservation.try_resize(self.store.bytes)
    }

    /// Load the next non-empty right batch. `pending` are the live refs of the
    /// left batch being processed (rewritten if the store compacts).
    /// Returns false at the end of the right input.
    async fn load_right(
        &mut self,
        target: Option<&Key<TN, G>>,
        pending: &mut [u64],
    ) -> Result<bool> {
        self.rkeys = None;
        loop {
            if let Some(t) = target {
                self.maybe_jump(t)?;
            }
            let Some(s) = self.stream.as_mut() else {
                return Ok(false);
            };
            let next = s.next().await;
            match next {
                None => {
                    self.stream = None;
                    if self.once
                        && let (Some(last), Some(next)) = (&self.right_last, &self.next_first)
                        && cmp_keys::<TN, G>(last, next) == Ordering::Greater
                    {
                        return exec_err!(
                            "StreamingAsOfJoin: right input partitions are not consecutive \
                             ranges of the join key order (a partition ends after the next \
                             one starts)"
                        );
                    }
                    let n = self.pro.right_order.len();
                    if self.cur_idx + 1 < n {
                        self.open(self.cur_idx + 1)?;
                        continue;
                    }
                    return Ok(false);
                }
                Some(b) => {
                    let b = b?;
                    let elapsed = self.elapsed.clone();
                    let t = elapsed.timer();
                    let prepared = right_keys::<TN, G>(
                        &*self.group,
                        &self.spec.right_ts,
                        &self.spec.right_by,
                        &self.spec.payload_idx,
                        &b,
                    )?;
                    let Some((keys, payload)) = prepared else {
                        continue;
                    };
                    self.check_right_order(&keys)?;
                    if self.store.needs_compaction() {
                        self.compact(pending)?;
                    }
                    let src = self.store.push(payload);
                    self.rkeys = Some(keys);
                    self.rpos = 0;
                    self.rsrc = src;
                    t.done();
                    self.update_reservation()?;
                    return Ok(true);
                }
            }
        }
    }

    // ----- the kernel ---------------------------------------------------------

    /// Answer left rows `from..` until the current right batch is exhausted.
    /// Returns the first unanswered row (== `lk.len` when done).
    #[inline]
    fn kernel(&mut self, lk: &Keys<TN, G>, from: usize, refs: &mut [u64]) -> usize {
        match (G::PACKS, lk.valid.is_none()) {
            (true, true) => self.kernel_impl::<true, false>(lk, from, refs),
            (true, false) => self.kernel_impl::<true, true>(lk, from, refs),
            (false, true) => self.kernel_impl::<false, false>(lk, from, refs),
            (false, false) => self.kernel_impl::<false, true>(lk, from, refs),
        }
    }

    /// The merge loop. For each left row, gallop the right cursor to the last
    /// right row `<=` it in `(group, ts)` order (`<` when strict) and take
    /// that row if it is in the same group. `PACKED`: keys are order-
    /// preserving i128s `(group, ts)` (no `by` = group 0), so a comparison is
    /// one integer compare; otherwise group columns are compared row by row
    /// (multi-column or non-primitive `by`).
    fn kernel_impl<const PACKED: bool, const NULLS: bool>(
        &mut self,
        lk: &Keys<TN, G>,
        from: usize,
        refs: &mut [u64],
    ) -> usize {
        let strict = self.spec.strict;
        let g: &G = &self.group;
        let lp: &[i128] = if PACKED { lk.packed.as_deref().expect("packed left keys") } else { &[] };
        let lts = lk.ts.as_slice();
        let n = lk.len;
        let mut i = from;
        while i < n {
            if NULLS && !lk.is_valid(i) {
                refs[i] = NONE;
                i += 1;
                continue;
            }
            if !self.right_eof {
                let Some(rk) = self.rkeys.as_ref() else {
                    return i;
                };
                let rp: &[i128] = if PACKED { rk.packed.as_deref().expect("packed right keys") } else { &[] };
                let rts = rk.ts.as_slice();
                let (rlen, pos) = (rk.len, self.rpos);
                // Is right row j before left row i (<=, or < when strict)?
                let le = |j: usize| -> bool {
                    let ord = if PACKED {
                        rp[j].cmp(&lp[i])
                    } else {
                        g.cmp_at(&rk.grp, j, &lk.grp, i).then(rts[j].cmp(&lts[i]))
                    };
                    if strict { ord == Ordering::Less } else { ord != Ordering::Greater }
                };
                // O(1) whole-batch skip on the last key, else gallop.
                let e = if pos < rlen && le(rlen - 1) { rlen } else { gallop_by(rlen, pos, le) };
                if e > pos {
                    // Ties: e - 1 is the LAST row <= key (upper bound - 1).
                    self.cand = make_ref(self.rsrc, e - 1);
                    if PACKED {
                        self.cand_gpk = packed_group(rp[e - 1]);
                    } else {
                        self.cand_group = Some(g.key_owned(&rk.grp, e - 1));
                    }
                }
                self.rpos = e;
                if e == rlen {
                    return i;
                }
            }
            let same_group = if PACKED {
                self.cand_gpk == packed_group(lp[i])
            } else {
                self.cand_group
                    .as_ref()
                    .is_some_and(|cg| g.cmp_owned(cg, &lk.grp, i) == Ordering::Equal)
            };
            refs[i] = if self.cand != NONE && same_group { self.cand } else { NONE };
            i += 1;
        }
        n
    }

    async fn run_single(
        &mut self,
        mut peeked: VecDeque<RecordBatch>,
        mut stream: Option<SendableRecordBatchStream>,
    ) -> Result<()> {
        let first = match &self.first_override {
            Some(f) => f.clone(),
            None if self.once => None,
            None => self.pro.left_first[self.p].clone(),
        };
        match first {
            Some(first) => {
                let idx = self.find_start(&first, self.spec.strict);
                self.open(idx)?;
            }
            // Execute-once: the unit is the right input.
            None if self.once => self.open(0)?,
            // No valid left key: every row is unmatched, no right input needed.
            None => self.right_eof = true,
        }
        while let Some(batch) = next_left(&mut peeked, &mut stream).await? {
            if batch.num_rows() == 0 {
                continue;
            }
            let elapsed = self.elapsed.clone();
            let t = elapsed.timer();
            let lk = left_keys::<TN, G>(
                &*self.group,
                &self.spec.left_ts,
                &self.spec.left_by,
                &batch,
            )?;
            self.check_left_order(&lk)?;
            let n = lk.len;
            let mut refs = vec![NONE; n];
            let mut i = 0;
            t.done();
            loop {
                {
                    let _t = elapsed.timer();
                    i = self.kernel(&lk, i, &mut refs);
                }
                if i >= n {
                    break;
                }
                let target = lk.key_at(&*self.group, i);
                if !self.load_right(Some(&target), &mut refs).await? {
                    self.right_eof = true;
                }
            }
            let out = {
                let _t = elapsed.timer();
                build_output(
                    &self.spec.out_schema,
                    &self.spec.out_cols,
                    &batch,
                    &refs,
                    &self.store,
                )?
            };
            let rows = out.num_rows();
            if self.tx.send(Ok(out)).await.is_err() {
                return Ok(()); // receiver dropped
            }
            self.metrics.record_output(rows);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Execute-once mode
// ---------------------------------------------------------------------------
//
// For inputs that must not be executed twice (joins, aggregates,
// repartitions): every input partition is executed exactly once.
//
// * The left partitions are read in order as one stream (consecutive ranges
//   of the key order, or a single partition).
// * Right partitions are either consecutive, disjoint key ranges
//   (`right_units`): each is peeked once for its first key and the stream is
//   kept, and the non-empty ones, sorted by first key, are the *units*; or
//   they are read in order as one unit.
// * A left row with key k belongs to the last unit whose first key is <= k
//   (< k when strict), so its predecessor lies inside that unit.
// * A serial distributor reads the left stream, slices each batch at unit
//   boundaries (galloping on the keys) and sends the slices to the units'
//   workers over bounded channels. NULL-key rows stay with the unit of the
//   preceding valid row (unit 0 at the start).
// * One worker per unit runs the merge kernel on its one right stream.
// * The driver emits the units' outputs in unit order (= left order), with
//   at most `target_partitions + window_extra` units active.
//
// Memory: per active unit, a bounded channel of left slices and a bounded
// channel of output batches, plus the right store (O(groups + batch)); right
// units not yet started hold one peeked batch each.

/// Capacity (in batches) of each unit's left-slice and output channels.
const UNIT_CHANNEL: usize = 4;

struct OnceArgs<G: GroupKind> {
    spec: Arc<JoinSpec>,
    group: Arc<G>,
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    right_units: bool,
    ctx: Arc<TaskContext>,
    tx: Sender<Result<RecordBatch>>,
    metrics: BaselineMetrics,
}

/// A record batch stream of `peeked` followed by `rest`.
fn chain_stream(
    schema: SchemaRef,
    peeked: VecDeque<RecordBatch>,
    rest: Option<SendableRecordBatchStream>,
) -> SendableRecordBatchStream {
    let head = futures::stream::iter(peeked.into_iter().map(Ok));
    match rest {
        Some(rest) => Box::pin(RecordBatchStreamAdapter::new(schema, head.chain(rest))),
        None => Box::pin(RecordBatchStreamAdapter::new(schema, head)),
    }
}

/// All partitions of `plan`, in order, as one stream; each is executed once.
///
/// Every partition is first polled to its first batch concurrently: some
/// operators (a `CollectLeft` hash join's shared build, for example) need all
/// of their partitions to start before any one makes progress. Then the
/// partitions are streamed in order.
fn concat_partitions(plan: Arc<dyn ExecutionPlan>, ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
    use futures::TryStreamExt;
    let schema = plan.schema();
    let primed = async move {
        let n = plan.output_partitioning().partition_count();
        let mut set = tokio::task::JoinSet::new();
        for q in 0..n {
            let (p, c) = (Arc::clone(&plan), Arc::clone(&ctx));
            set.spawn(async move {
                let mut s = p.execute(q, c)?;
                let first = s.next().await.transpose()?;
                Result::<_>::Ok((q, first, s))
            });
        }
        let mut parts = Vec::with_capacity(n);
        while let Some(r) = set.join_next().await {
            parts.push(r.map_err(join_err)??);
        }
        parts.sort_by_key(|p| p.0);
        let streams = parts
            .into_iter()
            .map(|(_, first, rest)| futures::stream::iter(first.map(Ok)).chain(rest));
        Result::<_>::Ok(futures::stream::iter(streams).flatten())
    };
    let s = futures::stream::once(primed).try_flatten();
    Box::pin(RecordBatchStreamAdapter::new(schema, s))
}

/// A record batch stream fed by a channel.
fn receiver_stream(
    schema: SchemaRef,
    rx: tokio::sync::mpsc::Receiver<RecordBatch>,
) -> SendableRecordBatchStream {
    let s = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|b| (Ok(b), rx))
    });
    Box::pin(RecordBatchStreamAdapter::new(schema, s))
}

/// Execute right partition `q` once and read it up to its first usable key.
/// The batches read so far and the open stream are returned with the key.
async fn peek_right<TN: TsNative, G: GroupKind>(
    right: Arc<dyn ExecutionPlan>,
    q: usize,
    ctx: Arc<TaskContext>,
    spec: Arc<JoinSpec>,
    g: Arc<G>,
) -> Result<Option<(Key<TN, G>, LeftStart)>> {
    let mut s = right.execute(q, ctx)?;
    let mut batches = VecDeque::new();
    while let Some(b) = s.next().await {
        let b = b?;
        let first = right_keys::<TN, G>(&*g, &spec.right_ts, &spec.right_by, &[], &b)?
            .map(|(k, _)| k.key_at(&*g, 0));
        if b.num_rows() > 0 {
            batches.push_back(b);
        }
        if let Some(first) = first {
            return Ok(Some((first, (batches, Some(s)))));
        }
    }
    Ok(None)
}

/// Compare unit first key `f` with valid row `r` of `k`.
#[inline]
fn cmp_key_row<TN: TsNative, G: GroupKind>(g: &G, f: &Key<TN, G>, k: &Keys<TN, G>, r: usize) -> Ordering {
    g.cmp_owned(&f.0, &k.grp, r).then(f.1.cmp(&k.ts.as_slice()[r]))
}

/// The serial distributor: read the left stream once and send each unit its
/// slices. `ufirst` are the first keys of units 1.. (unit 0 takes everything
/// before unit 1).
async fn distribute<TN: TsNative, G: GroupKind>(
    mut s: SendableRecordBatchStream,
    spec: Arc<JoinSpec>,
    g: Arc<G>,
    ufirst: Vec<Key<TN, G>>,
    mut txs: Vec<Option<tokio::sync::mpsc::Sender<RecordBatch>>>,
) -> Result<()> {
    let strict = spec.strict;
    let nu = txs.len();
    let mut cur = 0usize;
    let mut last: Option<Key<TN, G>> = None;
    // Does valid row `r` belong to unit `u` or a later one?
    let passes = |u: usize, lk: &Keys<TN, G>, r: usize| -> bool {
        let o = cmp_key_row::<TN, G>(&g, &ufirst[u], lk, r);
        if strict { o == Ordering::Less } else { o != Ordering::Greater }
    };
    while let Some(b) = s.next().await {
        let b = b?;
        let n = b.num_rows();
        if n == 0 {
            continue;
        }
        let lk = left_keys::<TN, G>(&*g, &spec.left_ts, &spec.left_by, &b)?;
        if let Some(f) = lk.first_valid() {
            let fk = lk.key_at(&*g, f);
            if let Some(p) = &last
                && cmp_keys::<TN, G>(p, &fk) == Ordering::Greater
            {
                return exec_err!(
                    "left input is not sorted by {} (a batch starts before the previous one ended)",
                    spec.left_key_desc
                );
            }
            if spec.options.validate_left && !lk.is_sorted(&g) {
                return exec_err!("left input is not sorted by {}", spec.left_key_desc);
            }
            last = lk.last_valid().map(|l| lk.key_at(&*g, l));
        }
        let vi: Option<Vec<usize>> =
            lk.valid.as_ref().map(|_| (0..n).filter(|&i| lk.is_valid(i)).collect());
        let nv = vi.as_ref().map_or(n, |v| v.len());
        let row = |j: usize| vi.as_ref().map_or(j, |v| v[j]);
        let mut start = 0usize;
        let mut j = 0usize;
        loop {
            let end_j = if cur + 1 < nu {
                gallop_by(nv, j, |jj| !passes(cur + 1, &lk, row(jj)))
            } else {
                nv
            };
            let end_row = if end_j < nv { row(end_j) } else { n };
            if end_row > start {
                let Some(tx) = txs[cur].as_ref() else {
                    return exec_err!("StreamingAsOfJoin: internal error, unit {cur} closed");
                };
                if tx.send(b.slice(start, end_row - start)).await.is_err() {
                    return Ok(()); // the worker failed; it reports the error
                }
            }
            if end_j >= nv {
                break;
            }
            let r = row(end_j);
            while cur + 1 < nu && passes(cur + 1, &lk, r) {
                txs[cur] = None; // close: the unit has all its left rows
                cur += 1;
            }
            start = end_row;
            j = end_j;
        }
    }
    Ok(())
}

async fn run_once<TN: TsNative, G: GroupKind>(a: OnceArgs<G>) -> Result<()> {
    let threads = a.ctx.session_config().target_partitions().max(1);
    let window = threads + a.spec.options.window_extra.max(1);

    // The units: each right partition peeked once (the probe), or all right
    // partitions in order as one unit.
    let mut units: Vec<(Option<Key<TN, G>>, LeftStart)> = Vec::new();
    if a.right_units {
        let mut set = tokio::task::JoinSet::new();
        for q in 0..a.right.output_partitioning().partition_count() {
            let (r, c, s, g) = (Arc::clone(&a.right), Arc::clone(&a.ctx), Arc::clone(&a.spec), Arc::clone(&a.group));
            set.spawn(async move { (q, peek_right::<TN, G>(r, q, c, s, g).await) });
        }
        let mut found: Vec<(Key<TN, G>, usize, LeftStart)> = Vec::new();
        while let Some(res) = set.join_next().await {
            let (q, r) = res.map_err(join_err)?;
            if let Some((k, start)) = r? {
                found.push((k, q, start));
            }
        }
        found.sort_by(|x, y| cmp_keys::<TN, G>(&x.0, &y.0).then(x.1.cmp(&y.1)));
        units = found.into_iter().map(|(k, _, s)| (Some(k), s)).collect();
    }
    if units.is_empty() {
        // One unit: every right partition in order (none if right_units
        // found no rows).
        let rest = (!a.right_units).then(|| concat_partitions(Arc::clone(&a.right), Arc::clone(&a.ctx)));
        units.push((None, (VecDeque::new(), rest)));
    }
    let n_units = units.len();
    let first_keys: Vec<Option<Key<TN, G>>> = units.iter().map(|u| u.0.clone()).collect();
    let ufirst: Vec<Key<TN, G>> = first_keys.iter().flatten().cloned().collect();

    let mut left_txs = Vec::with_capacity(n_units);
    type Pending = Option<(tokio::sync::mpsc::Receiver<RecordBatch>, LeftStart)>;
    let mut pending: Vec<Pending> = Vec::with_capacity(n_units);
    for (_, start) in units {
        let (tx, rx) = tokio::sync::mpsc::channel(UNIT_CHANNEL);
        left_txs.push(Some(tx));
        pending.push(Some((rx, start)));
    }
    let dist = tokio::spawn(distribute::<TN, G>(
        concat_partitions(Arc::clone(&a.left), Arc::clone(&a.ctx)),
        Arc::clone(&a.spec),
        Arc::clone(&a.group),
        ufirst,
        left_txs,
    ));
    // Aborts the distributor if the consumer goes away early.
    struct AbortOnDrop(tokio::task::AbortHandle);
    impl Drop for AbortOnDrop {
        fn drop(&mut self) {
            self.0.abort();
        }
    }
    let _dist_guard = AbortOnDrop(dist.abort_handle());

    let left_schema = a.left.schema();
    let mut outs: VecDeque<(tokio::sync::mpsc::Receiver<Result<RecordBatch>>, tokio::task::JoinHandle<()>)> =
        VecDeque::new();
    let mut launched = 0usize;
    let launch = |u: usize, pending: &mut Vec<Pending>| {
        let (rx, start) = pending[u].take().expect("unit launched once");
        let (otx, orx) = tokio::sync::mpsc::channel(UNIT_CHANNEL);
        let pro = Arc::new(Prologue::<TN, G> {
            left: Mutex::new(Vec::new()),
            left_first: Vec::new(),
            right_order: vec![0],
            right_first: first_keys[u].iter().cloned().collect(),
        });
        let reservation = MemoryConsumer::new(format!("StreamingAsOfJoin[unit {u}]"))
            .register(a.ctx.memory_pool());
        let mut w = new_worker(
            u,
            pro,
            Arc::clone(&a.spec),
            Arc::clone(&a.group),
            Arc::clone(&a.right),
            Arc::clone(&a.ctx),
            otx,
            a.metrics.clone(),
            reservation,
            None,
        );
        w.once = true;
        w.unit = Some(start);
        w.next_first = first_keys.get(u + 1).cloned().flatten();
        let left_stream = receiver_stream(Arc::clone(&left_schema), rx);
        let h = tokio::spawn(async move {
            let r = w.run_single(VecDeque::new(), Some(left_stream)).await;
            w.reservation.free();
            if let Err(e) = r {
                let _ = w.tx.send(Err(e)).await;
            }
        });
        (orx, h)
    };
    for e in 0..n_units {
        while launched < n_units && launched < e + window {
            outs.push_back(launch(launched, &mut pending));
            launched += 1;
        }
        let (mut rx, h) = outs.pop_front().expect("unit launched");
        while let Some(b) = rx.recv().await {
            let failed = b.is_err();
            if a.tx.send(b).await.is_err() || failed {
                return Ok(()); // consumer gone, or the error was forwarded
            }
        }
        h.await.map_err(join_err)?;
    }
    dist.await.map_err(join_err)??;
    a.metrics.done();
    Ok(())
}
