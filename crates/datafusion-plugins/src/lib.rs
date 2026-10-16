//! Streaming, range-parallel backward ASOF join for Apache DataFusion 55,
//! with polars `join_asof(strategy="backward")` semantics.
//!
//! * [`JoinAsof`]: build the join over two physical plans. Inputs must be
//!   sorted by `(by..., on)` (or `on`); nothing is sorted for you.
//! * [`AsofPlanTable`]: the join of two datafusion-proto serialized logical
//!   plans, as a table provider (how DataFrames from another DataFusion
//!   instance, such as datafusion-python, become inputs).
//! * [`StreamingAsOfJoinExec`], [`OrderedConcatExec`], [`AsofJoinScanExec`]:
//!   the operators.
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

mod exec;
mod kernel;
mod keys;
mod ordered;
mod plan;
mod plan_bytes;

pub use exec::{AsofOptions, StreamingAsOfJoinExec};
pub use ordered::OrderedConcatExec;
pub use plan::{AsofJoinScanExec, JoinAsof, OutputColumn, Side, last_plan};
pub use plan_bytes::AsofPlanTable;
