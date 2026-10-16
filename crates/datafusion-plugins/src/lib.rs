//! Custom operators for Apache DataFusion 55, usable from Rust and (through
//! the datafusion-python FFI) from Python.
//!
//! Nodes:
//! * [`JoinAsof`]: streaming backward ASOF join with polars `join_asof`
//!   semantics.
//! * [`JoinAsofMarkouts`]: ASOF joins at several time offsets ("markouts")
//!   in one pass over the right input.
//!
//! Both plug in after any upstream plan: each input partition is executed
//! once, the partitions are merged in key order ([`SortedMergeExec`] is the
//! same merge as a standalone operator), and the order is either required
//! (DataFusion sorts only where it can't prove it) or assumed and checked.
//!
//! Each node implements [`PlanNode`] (physical input plans in, physical plan
//! out). [`StreamInputTable`] runs a node over inputs another engine executes
//! (the Python package: datafusion-python DataFrames, read as Arrow
//! streams); [`PlanBytesTable`] over serialized logical plans planned here.
//! [`BridgeExec`] is the leaf a node's plan is handed out in.
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

mod asof;
mod bridge;
mod input;
mod join_asof;
mod kernel;
mod keys;
mod markouts;
mod merge;
mod plan_bytes;

pub use asof::{AsofExec, AsofSpec};
pub use bridge::{BridgeExec, last_plan, set_memory_pool};
pub use input::{BatchReader, HostInputExec, PartitionSource, PreparedSource, StreamInputTable};
pub use join_asof::{JoinAsof, OutputColumn, Side};
pub use markouts::{JoinAsofMarkouts, Markout};
pub use merge::SortedMergeExec;
pub use plan_bytes::{PlanBytesTable, PlanNode};
