//! Custom operators for Apache DataFusion 55, usable from Rust and (through
//! the datafusion-python FFI) from Python.
//!
//! Nodes:
//! * [`JoinAsof`]: streaming backward ASOF join with polars `join_asof`
//!   semantics, optionally at several time offsets ("markouts") in one pass
//!   over the right input.
//!
//! A node plugs in after any upstream plan: each input partition is executed
//! once, the partitions are merged in key order ([`SortedMergeExec`] is the
//! same merge as a standalone operator), and the order is either required
//! (DataFusion sorts only where it can't prove it) or assumed and checked.
//!
//! Each node implements [`PlanNode`] (physical input plans in, physical plan
//! out). [`NodeTable`] runs a node as a table provider, over serialized
//! logical plans planned here ([`PlanInputs`]) or inputs another engine
//! executes ([`HostInputs`]: the Python package's datafusion-python
//! DataFrames, read as Arrow streams).
//!
//! ```no_run
//! # async fn example() -> datafusion::common::Result<()> {
//! use datafusion::prelude::*;
//! use datafusion_plugins::JoinAsof;
//!
//! let ctx = SessionContext::new();
//! let order = vec![vec![col("security_id").sort(true, true), col("ts").sort(true, true)]];
//! let trades = ctx
//!     .read_parquet("trades.parquet", ParquetReadOptions::default().file_sort_order(order.clone()))
//!     .await?;
//! let quotes = ctx
//!     .read_parquet("quotes.parquet", ParquetReadOptions::default().file_sort_order(order))
//!     .await?;
//! let plan = JoinAsof::new("ts").by(["security_id"]).build(
//!     trades.create_physical_plan().await?,
//!     quotes.create_physical_plan().await?,
//!     ctx.state().config_options(),
//! )?;
//! let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;
//! # Ok(()) }
//! ```

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::Result;
use datafusion::common::config::ConfigOptions;
use datafusion::physical_plan::ExecutionPlan;

mod asof;
mod bridge;
mod input;
mod join_asof;
mod kernel;
mod keys;
mod merge;
mod ordered;
mod table;

pub use input::{BatchReader, HostInputs, PartitionSource, PreparedSource};
pub use join_asof::{JoinAsof, Markout, OutputColumn, Side};
pub use merge::SortedMergeExec;
pub use table::{NodeInputs, NodeTable, PlanInputs};

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
    fn required_columns(&self, inputs: &[SchemaRef], _projection: &[usize]) -> Result<Vec<Option<Vec<String>>>> {
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
