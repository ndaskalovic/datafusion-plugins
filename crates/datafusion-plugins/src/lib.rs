//! Custom operators for Apache DataFusion 55, usable from Rust and (through
//! the datafusion-python FFI) from Python.
//!
//! Nodes:
//! * [`JoinAsof`]: streaming backward ASOF join with polars `join_asof`
//!   semantics.
//! * [`JoinAsofMarkouts`]: ASOF joins at several time offsets ("markouts")
//!   in one pass over the right input.
//!
//! Each node implements [`PlanNode`] (physical input plans in, physical plan
//! out), so [`PlanBytesTable`] can run it over serialized logical plans, the
//! way the Python package does. [`BridgeExec`] is the leaf the plan is handed
//! out in; [`OrderedConcatExec`] is a reusable building block.
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

mod bridge;
mod exec;
mod join_asof;
mod kernel;
mod keys;
mod markouts;
mod ordered;
mod plan_bytes;
mod scan;

pub use bridge::{BridgeExec, last_plan};
pub use exec::{AsofOptions, StreamingAsOfJoinExec};
pub use join_asof::{JoinAsof, OutputColumn, Side};
pub use markouts::{JoinAsofMarkouts, Markout, MarkoutsExec};
pub use ordered::OrderedConcatExec;
pub use plan_bytes::{PlanBytesTable, PlanNode};
