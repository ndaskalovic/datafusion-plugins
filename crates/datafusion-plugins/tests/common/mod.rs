//! Helpers shared by the integration tests.
#![allow(dead_code)]

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{DataType, Int64Type, SchemaRef};
use datafusion::common::Result;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::execution::TaskContext;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::WriterProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::{PartitionStream, StreamingTableExec};
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use datafusion::prelude::{SessionConfig, SessionContext};

/// xorshift64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// Small batches and many scan ranges, so tiny files split and stream.
pub fn context(target_partitions: usize, batch_size: usize) -> SessionContext {
    let config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_batch_size(batch_size)
        .set_usize("datafusion.optimizer.repartition_file_min_size", 1)
        .set_bool("datafusion.execution.enable_file_stream_work_stealing", false);
    SessionContext::new_with_config(config)
}

pub fn write_parquet(path: &Path, batch: &RecordBatch, row_group: usize) {
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(row_group)).build();
    let mut w = ArrowWriter::try_new(File::create(path).unwrap(), batch.schema(), Some(props)).unwrap();
    w.write(batch).unwrap();
    w.close().unwrap();
}

/// In-memory partitions (no declared order).
pub fn mem(schema: &SchemaRef, parts: Vec<Vec<RecordBatch>>) -> Arc<dyn ExecutionPlan> {
    MemorySourceConfig::try_new_exec(&parts, Arc::clone(schema), None).unwrap()
}

/// One partition's batches; `fail_at = Some(k)` panics (or errors, with
/// `error`) instead of producing batch `k`.
#[derive(Debug)]
struct Part {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    fail_at: Option<usize>,
    error: bool,
}

impl PartitionStream for Part {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }
    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let (fail_at, error) = (self.fail_at, self.error);
        let items = self.batches.clone().into_iter().enumerate().map(move |(k, b)| -> Result<RecordBatch> {
            if fail_at == Some(k) {
                if error {
                    return datafusion::common::exec_err!("upstream failed at batch {k}");
                }
                panic!("upstream panicked at batch {k}");
            }
            Ok(b)
        });
        Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), futures::stream::iter(items)))
    }
}

/// Partitions of batches where partition `p` fails at batch `k` for
/// `fail = Some((p, k, error))`.
pub fn failing(schema: &SchemaRef, parts: Vec<Vec<RecordBatch>>, fail: Option<(usize, usize, bool)>) -> Arc<dyn ExecutionPlan> {
    let parts: Vec<Arc<dyn PartitionStream>> = parts
        .into_iter()
        .enumerate()
        .map(|(p, batches)| {
            let (fail_at, error) = match fail {
                Some((q, k, e)) if q == p => (Some(k), e),
                _ => (None, false),
            };
            Arc::new(Part { schema: Arc::clone(schema), batches, fail_at, error }) as _
        })
        .collect();
    Arc::new(StreamingTableExec::try_new(Arc::clone(schema), parts, None, [], false, None).unwrap())
}

/// Column `name` of `batches` as optional i64s (any integer-like type).
pub fn ints(batches: &[RecordBatch], name: &str) -> Vec<Option<i64>> {
    let mut out = Vec::new();
    for b in batches {
        let c = b.column_by_name(name).unwrap_or_else(|| panic!("no column {name} in {:?}", b.schema()));
        let c = arrow::compute::cast(c, &DataType::Int64).unwrap();
        let c = c.as_primitive::<Int64Type>();
        out.extend((0..c.len()).map(|i| c.is_valid(i).then(|| c.value(i))));
    }
    out
}
