//! Shared by nodes over sorted inputs.
//!
//! A node either *requires* each input partition sorted by its keys
//! (ascending; DataFusion's `EnsureRequirements` adds a sort only where it
//! can't prove the order) or, with `assume_sorted`, trusts the order and
//! checks it while streaming. The NULL placement follows the input when it
//! is provable, so nothing is re-sorted for it.

use std::sync::Arc;

use arrow::compute::SortOptions;
use datafusion::common::Result;
use datafusion::common::config::ConfigOptions;
use datafusion::physical_expr::{EquivalenceProperties, LexOrdering, OrderingRequirements, PhysicalExprRef, PhysicalSortExpr};
use datafusion::physical_expr_common::physical_expr::fmt_sql;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::ensure_requirements::EnsureRequirements;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use crate::merge::OrderCheck;

/// Ascending sort expressions on `exprs`.
pub(crate) fn sort_exprs(exprs: &[PhysicalExprRef], nulls_first: bool) -> Vec<PhysicalSortExpr> {
    let options = SortOptions { descending: false, nulls_first };
    exprs.iter().map(|e| PhysicalSortExpr { expr: Arc::clone(e), options }).collect()
}

/// The requirement "sorted by `exprs`" (`None` for no keys).
pub(crate) fn requirement(exprs: &[PhysicalExprRef], nulls_first: bool) -> Option<OrderingRequirements> {
    LexOrdering::new(sort_exprs(exprs, nulls_first)).map(OrderingRequirements::from)
}

/// `Some(nulls_first)` if `eq` satisfies an ascending order on `exprs`.
fn nulls_first_of(eq: &EquivalenceProperties, exprs: &[PhysicalExprRef]) -> Result<Option<bool>> {
    for nf in [true, false] {
        if eq.ordering_satisfy(sort_exprs(exprs, nf))? {
            return Ok(Some(nf));
        }
    }
    Ok(None)
}

/// The NULL placement of `input`'s order on `exprs`: the provable one, else
/// NULLS FIRST when the order is required, `None` (unknown) when assumed.
pub(crate) fn nulls_first(input: &Arc<dyn ExecutionPlan>, exprs: &[PhysicalExprRef], assume_sorted: bool) -> Result<Option<bool>> {
    let p = nulls_first_of(input.equivalence_properties(), exprs)?;
    Ok(if assume_sorted { p } else { Some(p.unwrap_or(true)) })
}

/// `(a, b)`, for messages.
pub(crate) fn key_desc(exprs: &[PhysicalExprRef]) -> String {
    let names: Vec<String> = exprs.iter().map(|e| fmt_sql(e.as_ref()).to_string()).collect();
    format!("({})", names.join(", "))
}

/// How a merge checks one input's partitions: at batch boundaries, and with
/// `assume_sorted` and `check_sortedness` inside every batch.
pub(crate) fn order_check(side: &str, exprs: &[PhysicalExprRef], assume_sorted: bool, check_sortedness: bool) -> OrderCheck {
    OrderCheck {
        message: format!("{side} input is not sorted by {}", key_desc(exprs)),
        in_batch: assume_sorted && check_sortedness,
        hint: if assume_sorted {
            "; each input partition must arrive sorted with assume_sorted=True (sort it upstream, or pass \
             assume_sorted=False to let the node sort)"
                .to_string()
        } else {
            String::new()
        },
    }
}

/// Let DataFusion add the sorts `plan`'s inputs require, unless the order is
/// assumed.
pub(crate) fn enforce(plan: Arc<dyn ExecutionPlan>, assume_sorted: bool, config: &ConfigOptions) -> Result<Arc<dyn ExecutionPlan>> {
    if assume_sorted { Ok(plan) } else { EnsureRequirements::new().optimize(plan, config) }
}
