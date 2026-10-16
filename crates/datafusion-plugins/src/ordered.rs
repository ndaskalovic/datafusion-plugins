//! `OrderedConcatExec`: one ordered output partition from N input partitions
//! that are consecutive, disjoint ranges of one sort order (byte-range splits
//! of a sorted file, and so the asof join's per-left-partition outputs).
//!
//! DataFusion would otherwise put a `SortPreservingMergeExec` here. On disjoint
//! ranges that merge drains partition 0 before it reads partition 1, so the
//! partitions it feeds from run one after another, and it compares every row.
//! This instead runs a sliding window of `window` partitions concurrently,
//! buffers each one's output, and emits them in partition order. Memory is
//! bounded by `window` partitions' output, not by the input.
//!
//! The order across partitions is checked at every boundary (last row of one
//! partition vs first row of the next, on the output ordering), so a wrong
//! assumption is an error, not a silently unsorted result.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::row::{OwnedRow, RowConverter, SortField};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Result, exec_err, plan_err};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{LexOrdering, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::projection::{ProjectionExec, make_with_child};
use datafusion::physical_plan::stream::RecordBatchReceiverStreamBuilder;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan,
    ExecutionPlanProperties, InputDistributionRequirements, Partitioning, PlanProperties,
    ReplaceChildrenOptions, SendableRecordBatchStream,
};
use futures::StreamExt;

#[derive(Debug)]
pub struct OrderedConcatExec {
    input: Arc<dyn ExecutionPlan>,
    ordering: LexOrdering,
    window: usize,
    cache: Arc<PlanProperties>,
}

impl OrderedConcatExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>, window: usize) -> Result<Self> {
        let Some(ordering) = input.equivalence_properties().output_ordering() else {
            return plan_err!("OrderedConcatExec: input has no output ordering");
        };
        let cache = PlanProperties::new(
            input.equivalence_properties().clone(),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Self {
            input,
            ordering,
            window: window.max(1),
            cache: Arc::new(cache),
        })
    }
}

impl DisplayAs for OrderedConcatExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "OrderedConcatExec: window={}, order=[{}]", self.window, self.ordering)
    }
}

impl ExecutionPlan for OrderedConcatExec {
    fn name(&self) -> &str {
        "OrderedConcatExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    #[allow(deprecated)]
    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.input_distribution_requirements().into_per_child()
    }

    fn input_distribution_requirements(&self) -> InputDistributionRequirements {
        InputDistributionRequirements::new(vec![Distribution::UnspecifiedDistribution])
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return plan_err!("OrderedConcatExec expects one child");
        }
        let input = children.swap_remove(0);
        match options.children_properties {
            ChildrenPropertiesMode::Keep => Ok(Arc::new(Self {
                input,
                ordering: self.ordering.clone(),
                window: self.window,
                cache: Arc::clone(&self.cache),
            })),
            ChildrenPropertiesMode::Recompute => Ok(Arc::new(Self::try_new(input, self.window)?)),
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

    /// Let a narrowing projection through, so it can be embedded in the join
    /// below (fewer right columns gathered). The order is recomputed from the
    /// new child; if the projection drops it, stay put.
    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if projection.expr().len() >= projection.input().schema().fields().len() {
            return Ok(None);
        }
        let child = make_with_child(projection, &self.input)?;
        Ok(Self::try_new(child, self.window)
            .ok()
            .map(|e| Arc::new(e) as Arc<dyn ExecutionPlan>))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return exec_err!("OrderedConcatExec has one output partition, got {partition}");
        }
        let input = Arc::clone(&self.input);
        let window = self.window;
        let ordering = self.ordering.clone();
        let mut builder = RecordBatchReceiverStreamBuilder::new(self.schema(), 2);
        let tx = builder.tx();
        builder.spawn(async move {
            let n = input.output_partitioning().partition_count();
            let spawn = |i: usize| {
                let (input, ctx) = (Arc::clone(&input), Arc::clone(&context));
                tokio::spawn(async move {
                    let mut s = input.execute(i, ctx)?;
                    let mut out = Vec::new();
                    while let Some(b) = s.next().await {
                        let b = b?;
                        if b.num_rows() > 0 {
                            out.push(b);
                        }
                    }
                    Result::<Vec<RecordBatch>>::Ok(out)
                })
            };
            let mut inflight = VecDeque::new();
            let mut next = 0;
            while next < n.min(window) {
                inflight.push_back(spawn(next));
                next += 1;
            }
            let mut check = BoundaryCheck::try_new(&ordering, &input.schema())?;
            while let Some(h) = inflight.pop_front() {
                let batches = h
                    .await
                    .map_err(|e| datafusion::common::DataFusionError::External(Box::new(e)))??;
                if next < n {
                    inflight.push_back(spawn(next));
                    next += 1;
                }
                if let (Some(first), Some(last)) = (batches.first(), batches.last()) {
                    check.step(first, last)?;
                }
                for b in batches {
                    if tx.send(Ok(b)).await.is_err() {
                        return Ok(()); // consumer dropped
                    }
                }
            }
            Ok(())
        });
        Ok(builder.build())
    }
}

/// Checks `last row of previous partition <= first row of this one`.
struct BoundaryCheck {
    ordering: LexOrdering,
    conv: RowConverter,
    prev_last: Option<OwnedRow>,
}

impl BoundaryCheck {
    fn try_new(ordering: &LexOrdering, schema: &arrow::datatypes::SchemaRef) -> Result<Self> {
        let fields = ordering
            .iter()
            .map(|e| Ok(SortField::new_with_options(e.expr.data_type(schema)?, e.options)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            ordering: ordering.clone(),
            conv: RowConverter::new(fields)?,
            prev_last: None,
        })
    }

    fn row(&self, b: &RecordBatch, i: usize) -> Result<OwnedRow> {
        let one = b.slice(i, 1);
        let cols = self
            .ordering
            .iter()
            .map(|e| e.expr.evaluate(&one)?.into_array(1))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.conv.convert_columns(&cols)?.row(0).owned())
    }

    fn step(&mut self, first: &RecordBatch, last: &RecordBatch) -> Result<()> {
        let f = self.row(first, 0)?;
        if let Some(p) = &self.prev_last
            && p.row() > f.row()
        {
            return exec_err!(
                "OrderedConcatExec: input partitions are not consecutive ranges of the \
                 declared order"
            );
        }
        self.prev_last = Some(self.row(last, last.num_rows() - 1)?);
        Ok(())
    }
}
