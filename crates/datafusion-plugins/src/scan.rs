//! Helpers shared by the nodes: recognising and splitting ordered scans,
//! and order checks on physical plans.

use std::sync::Arc;

use arrow::compute::SortOptions;
use datafusion::common::config::ConfigOptions;
use datafusion::common::{JoinType, Result, plan_datafusion_err};
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_expr::{EquivalenceProperties, LexOrdering, PhysicalExprRef, PhysicalSortExpr};
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, ReplaceChildrenOptions,
};

pub(crate) fn sort_exprs(exprs: &[PhysicalExprRef], nulls_first: bool) -> Vec<PhysicalSortExpr> {
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

/// `Some(nulls_first)` if `eq` satisfies an ascending order on `exprs`,
/// trying NULLS FIRST then NULLS LAST (null keys never match, so either
/// placement works).
pub(crate) fn nulls_first_of(eq: &EquivalenceProperties, exprs: &[PhysicalExprRef]) -> Result<Option<bool>> {
    for nf in [true, false] {
        if eq.ordering_satisfy(sort_exprs(exprs, nf))? {
            return Ok(Some(nf));
        }
    }
    Ok(None)
}

pub(crate) fn satisfies(plan: &Arc<dyn ExecutionPlan>, sort: &[PhysicalSortExpr]) -> Result<bool> {
    plan.equivalence_properties().ordering_satisfy(sort.to_vec())
}

/// Merge the (each sorted) partitions of `plan` into one ordered partition.
pub(crate) fn merge(plan: Arc<dyn ExecutionPlan>, sort: &[PhysicalSortExpr]) -> Result<Arc<dyn ExecutionPlan>> {
    if plan.output_partitioning().partition_count() <= 1 {
        return Ok(plan);
    }
    let lex = LexOrdering::new(sort.to_vec())
        .ok_or_else(|| plan_datafusion_err!("join_asof: empty sort key"))?;
    Ok(Arc::new(SortPreservingMergeExec::new(lex, plan)))
}

/// Nodes that keep their input's partitions and row order.
pub(crate) fn passthrough(plan: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    if plan.is::<FilterExec>()
        || plan.is::<ProjectionExec>()
        || plan.is::<CooperativeExec>()
    {
        plan.children().into_iter().next()
    } else {
        None
    }
}

/// The distinct files of a Parquet `DataSourceExec`, if `plan` is one.
pub(crate) fn parquet_files(plan: &Arc<dyn ExecutionPlan>) -> Option<Vec<String>> {
    let dse = plan.downcast_ref::<DataSourceExec>()?;
    let (conf, _) = dse.downcast_to_file_source::<ParquetSource>()?;
    let mut files: Vec<String> = Vec::new();
    for g in &conf.file_groups {
        for f in g.iter() {
            let p = f.object_meta.location.to_string();
            if !files.contains(&p) {
                files.push(p);
            }
        }
    }
    Some(files)
}

/// A re-executable scan of exactly one Parquet file, possibly under filters
/// and projections. Its byte-range partitions are consecutive key ranges.
pub(crate) fn single_file_scan(plan: &Arc<dyn ExecutionPlan>) -> bool {
    match passthrough(plan) {
        Some(child) => single_file_scan(child),
        None => parquet_files(plan).is_some_and(|f| f.len() == 1),
    }
}

/// Partitions are consecutive key ranges: a single-file scan, through
/// order-preserving nodes and the probe side of a `CollectLeft` inner join.
pub(crate) fn range_partitioned(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if let Some(child) = passthrough(plan) {
        return range_partitioned(child);
    }
    if let Some(hj) = plan.downcast_ref::<HashJoinExec>()
        && *hj.partition_mode() == PartitionMode::CollectLeft
        && *hj.join_type() == JoinType::Inner
        && hj.maintains_input_order().get(1).copied().unwrap_or(false)
    {
        return range_partitioned(hj.right());
    }
    single_file_scan(plan)
}

/// Turn off file-stream work stealing in the scan under a range-partitioned
/// plan (DataFusion 55 lets sibling partitions of an unordered scan steal
/// each other's byte ranges, so a partition would no longer be one range).
pub(crate) fn pin_order(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    let opts = || ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute);
    if let Some(child) = passthrough(&plan) {
        let child = pin_order(Arc::clone(child))?;
        return Arc::clone(&plan).replace_children(vec![child], opts());
    }
    if let Some(hj) = plan.downcast_ref::<HashJoinExec>() {
        let (l, r) = (Arc::clone(hj.left()), pin_order(Arc::clone(hj.right()))?);
        return Arc::clone(&plan).replace_children(vec![l, r], opts());
    }
    if plan.is::<DataSourceExec>() {
        return Ok(plan.with_preserve_order(true).unwrap_or(plan));
    }
    Ok(plan)
}

/// Re-split the single-file Parquet scan under `plan` (through filters and
/// projections) into consecutive byte ranges of about `size / target` bytes.
/// For one file this is DataFusion's own `repartition_evenly_by_size`.
/// Returns `None` if `plan` is not such a scan or no split applies.
pub(crate) fn resplit(
    plan: &Arc<dyn ExecutionPlan>,
    target: usize,
    config: &ConfigOptions,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(child) = passthrough(plan) {
        let Some(new_child) = resplit(child, target, config)? else {
            return Ok(None);
        };
        let node = Arc::clone(plan).replace_children(
            vec![new_child],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )?;
        return Ok(Some(node));
    }
    let Some(dse) = plan.downcast_ref::<DataSourceExec>() else {
        return Ok(None);
    };
    let Some((conf, _)) = dse.downcast_to_file_source::<ParquetSource>() else {
        return Ok(None);
    };
    let Some(file) = conf.file_groups.iter().flat_map(|g| g.iter()).next() else {
        return Ok(None);
    };
    if parquet_files(plan).is_none_or(|f| f.len() != 1) {
        return Ok(None);
    }
    let size = file.object_meta.size;
    if !config.optimizer.repartition_file_scans
        || target <= 1
        || size < config.optimizer.repartition_file_min_size as u64
        || conf.file_compression_type.is_compressed()
    {
        return Ok(None);
    }
    let mut template = file.clone();
    template.range = None;
    let chunk = size.div_ceil(target as u64).max(1);
    let mut groups = Vec::new();
    let mut start = 0u64;
    while start < size {
        let end = size.min(start + chunk);
        groups.push(FileGroup::new(vec![template.clone().with_range(start as i64, end as i64)]));
        start = end;
    }
    if groups.len() <= conf.file_groups.len() {
        return Ok(None);
    }
    let new_conf = FileScanConfigBuilder::from(conf.clone())
        .with_file_groups(groups)
        .build();
    Ok(Some(DataSourceExec::from_data_source(new_conf) as Arc<dyn ExecutionPlan>))
}
