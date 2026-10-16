//! Batch-level building blocks: galloping search, right-row references, the
//! compacting right-row store and output assembly.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array, new_null_array,
};
use arrow::buffer::NullBuffer;
use arrow::compute::{filter, filter_record_batch, interleave, take};
use arrow::datatypes::SchemaRef;
use datafusion::common::Result;
use datafusion::physical_expr::PhysicalExprRef;

use crate::keys::{GroupKind, Keys, TsNative};

/// Reference to a right row: `(source batch << 32) | row`. `NONE` = no match.
pub const NONE: u64 = u64::MAX;

#[inline]
pub fn make_ref(src: u32, row: usize) -> u64 {
    ((src as u64) << 32) | row as u64
}

#[inline]
pub fn ref_src(r: u64) -> usize {
    (r >> 32) as usize
}

#[inline]
pub fn ref_row(r: u64) -> usize {
    (r & 0xFFFF_FFFF) as usize
}

/// First index in `[lo, hi)` where `le` is false (`le` monotone true..false).
#[inline]
fn bsearch<F: Fn(usize) -> bool>(mut lo: usize, mut hi: usize, le: &F) -> usize {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if le(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Galloping (exponential) search: first index `>= from` in `[from, n)` where
/// `le` is false, given `le` is monotone (true..false) on `[from, n)`.
///
/// O(log gap) where gap is the distance moved, so O(n) total over a merge of
/// sorted inputs and O(log n) for a single long skip.
#[inline]
pub fn gallop_by<F: Fn(usize) -> bool>(n: usize, from: usize, le: F) -> usize {
    if from >= n || !le(from) {
        return from;
    }
    // Invariant: le(lo) is true.
    let mut lo = from;
    let mut step = 1usize;
    loop {
        let hi = lo + step;
        if hi >= n {
            return bsearch(lo + 1, n, &le);
        }
        if !le(hi) {
            return bsearch(lo + 1, hi, &le);
        }
        lo = hi;
        step <<= 1;
    }
}

/// Gather rows `(batch, row)` from `batches` (all with `schema`) into one batch.
pub fn gather_from(
    schema: &SchemaRef,
    batches: &[&RecordBatch],
    idx: &[(usize, usize)],
) -> Result<RecordBatch> {
    if schema.fields().is_empty() {
        let opts = RecordBatchOptions::new().with_row_count(Some(idx.len()));
        return Ok(RecordBatch::try_new_with_options(
            Arc::clone(schema),
            vec![],
            &opts,
        )?);
    }
    if idx.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::clone(schema)));
    }
    let mut cols = Vec::with_capacity(schema.fields().len());
    for j in 0..schema.fields().len() {
        let arrs: Vec<&dyn Array> = batches.iter().map(|b| b.column(j).as_ref()).collect();
        cols.push(interleave(&arrs, idx)?);
    }
    Ok(RecordBatch::try_new(Arc::clone(schema), cols)?)
}

/// Holds the right batches (payload columns only) that live references point
/// into. When it grows past `max_sources` the referenced rows are copied into
/// one compact batch and everything else is dropped, so retained memory is
/// O(#groups + batch size) rather than O(right rows).
pub struct RightStore {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
    pub bytes: usize,
    pub max_sources: usize,
}

impl RightStore {
    pub fn new(schema: SchemaRef, max_sources: usize) -> Self {
        Self {
            schema,
            batches: Vec::new(),
            bytes: 0,
            max_sources,
        }
    }

    pub fn push(&mut self, b: RecordBatch) -> u32 {
        self.bytes += b.get_array_memory_size();
        self.batches.push(b);
        (self.batches.len() - 1) as u32
    }

    pub fn needs_compaction(&self) -> bool {
        self.batches.len() >= self.max_sources
    }

    /// Gather rows `(src, row)` into one batch with the store's schema.
    pub fn gather(&self, idx: &[(usize, usize)]) -> Result<RecordBatch> {
        let refs: Vec<&RecordBatch> = self.batches.iter().collect();
        gather_from(&self.schema, &refs, idx)
    }

    /// Rewrite every live reference in `sets` to point into a single compact
    /// batch (source 0) and drop all other batches.
    pub fn compact(&mut self, sets: &mut [&mut [u64]]) -> Result<()> {
        let mut idx: Vec<(usize, usize)> = Vec::new();
        let mut last_old = NONE;
        let mut last_new = NONE;
        for set in sets.iter_mut() {
            for r in set.iter_mut() {
                if *r == NONE {
                    continue;
                }
                if *r == last_old {
                    *r = last_new;
                    continue;
                }
                let old = *r;
                let new = make_ref(0, idx.len());
                idx.push((ref_src(old), ref_row(old)));
                last_old = old;
                last_new = new;
                *r = new;
            }
        }
        let compact = self.gather(&idx)?;
        self.batches.clear();
        self.bytes = 0;
        self.push(compact);
        Ok(())
    }

    /// Build one output column (payload column `j`) from references.
    ///
    /// `single_src` is the one source every non-NONE ref points into, if
    /// there is one (see [`single_source`]); then a plain `take` is used,
    /// otherwise `interleave`.
    pub fn take_column(
        &self,
        j: usize,
        refs: &[u64],
        single_src: Option<Option<usize>>,
    ) -> Result<ArrayRef> {
        let dt = self.schema.field(j).data_type();
        match single_src {
            // Every ref is NONE.
            Some(None) => return Ok(new_null_array(dt, refs.len())),
            Some(Some(src)) => {
                let idx: UInt32Array = refs
                    .iter()
                    .map(|r| if *r == NONE { None } else { Some(ref_row(*r) as u32) })
                    .collect();
                return Ok(take(self.batches[src].column(j).as_ref(), &idx, None)?);
            }
            None => {}
        }
        // Source 0 is a one-row null array used for unmatched rows.
        let null = new_null_array(dt, 1);
        let mut arrs: Vec<&dyn Array> = Vec::with_capacity(self.batches.len() + 1);
        arrs.push(null.as_ref());
        for b in &self.batches {
            arrs.push(b.column(j).as_ref());
        }
        let idx: Vec<(usize, usize)> = refs
            .iter()
            .map(|r| {
                if *r == NONE {
                    (0, 0)
                } else {
                    (ref_src(*r) + 1, ref_row(*r))
                }
            })
            .collect();
        Ok(interleave(&arrs, &idx)?)
    }
}

/// `Some(None)` if every ref is NONE, `Some(Some(src))` if all non-NONE refs
/// point into one source batch, `None` if they span several.
pub fn single_source(refs: &[u64]) -> Option<Option<usize>> {
    let mut src: Option<usize> = None;
    for r in refs {
        if *r == NONE {
            continue;
        }
        let s = ref_src(*r);
        match src {
            None => src = Some(s),
            Some(cur) if cur != s => return None,
            Some(_) => {}
        }
    }
    Some(src)
}

/// One output column: from the left batch, or from the right payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutCol {
    Left(usize),
    Right(usize),
}

/// Assemble an output batch: left columns are passed through (zero copy),
/// right columns are gathered from `store` by `refs`.
pub fn build_output(
    schema: &SchemaRef,
    out_cols: &[OutCol],
    left: &RecordBatch,
    refs: &[u64],
    store: &RightStore,
) -> Result<RecordBatch> {
    let mut cols = Vec::with_capacity(out_cols.len());
    let single = if out_cols.iter().any(|c| matches!(c, OutCol::Right(_))) {
        single_source(refs)
    } else {
        Some(None)
    };
    for c in out_cols {
        match c {
            OutCol::Left(i) => cols.push(Arc::clone(left.column(*i))),
            OutCol::Right(j) => cols.push(store.take_column(*j, refs, single)?),
        }
    }
    let opts = RecordBatchOptions::new().with_row_count(Some(left.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(schema),
        cols,
        &opts,
    )?)
}

/// Evaluate the ts and `by` expressions of one side and their combined
/// validity (`None` when every row is valid).
pub fn eval_side(
    ts_expr: &PhysicalExprRef,
    by: &[PhysicalExprRef],
    batch: &RecordBatch,
) -> Result<(ArrayRef, Vec<ArrayRef>, Option<NullBuffer>)> {
    let n = batch.num_rows();
    let ts = ts_expr.evaluate(batch)?.into_array(n)?;
    let by = by
        .iter()
        .map(|e| e.evaluate(batch)?.into_array(n))
        .collect::<Result<Vec<_>>>()?;
    let mut valid = ts.logical_nulls();
    for a in &by {
        valid = NullBuffer::union(valid.as_ref(), a.logical_nulls().as_ref());
    }
    let valid = valid.filter(|v| v.null_count() > 0);
    Ok((ts, by, valid))
}

/// Extract keys of a left batch (nulls kept, flagged in `valid`).
pub fn left_keys<TN: TsNative, G: GroupKind>(
    g: &G,
    ts_expr: &PhysicalExprRef,
    by: &[PhysicalExprRef],
    batch: &RecordBatch,
) -> Result<Keys<TN, G>> {
    let (ts, by, valid) = eval_side(ts_expr, by, batch)?;
    let (ts, grp, len) = (TN::extract(&ts)?, g.extract(&by)?, batch.num_rows());
    let packed = g.pack(&grp, ts.as_slice(), len);
    Ok(Keys { ts, grp, valid, len, packed })
}

/// Extract keys and payload of a right batch. Rows with a NULL ts or `by` key
/// can never match, so they are filtered out here. Returns `None` for a batch
/// with no usable rows.
pub fn right_keys<TN: TsNative, G: GroupKind>(
    g: &G,
    ts_expr: &PhysicalExprRef,
    by: &[PhysicalExprRef],
    payload_idx: &[usize],
    batch: &RecordBatch,
) -> Result<Option<(Keys<TN, G>, RecordBatch)>> {
    if batch.num_rows() == 0 {
        return Ok(None);
    }
    let (ts, by, valid) = eval_side(ts_expr, by, batch)?;
    let payload = batch.project(payload_idx)?;
    let (ts, by, payload) = match valid {
        Some(v) => {
            let mask = BooleanArray::new(v.inner().clone(), None);
            let ts = filter(ts.as_ref(), &mask)?;
            let by = by
                .iter()
                .map(|a| Ok(filter(a.as_ref(), &mask)?))
                .collect::<Result<Vec<_>>>()?;
            let payload = filter_record_batch(&payload, &mask)?;
            (ts, by, payload)
        }
        None => (ts, by, payload),
    };
    let len = ts.len();
    if len == 0 {
        return Ok(None);
    }
    let (ts, grp) = (TN::extract(&ts)?, g.extract(&by)?);
    let packed = g.pack(&grp, ts.as_slice(), len);
    Ok(Some((Keys { ts, grp, valid: None, len, packed }, payload)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gallop_matches_partition_point() {
        let v: Vec<i64> = vec![1, 1, 2, 3, 3, 3, 5, 8, 8, 9, 12, 12, 12, 20];
        for from in 0..=v.len() {
            for key in -1..22 {
                let expect = from + v[from..].partition_point(|x| *x <= key);
                let got = gallop_by(v.len(), from, |j| v[j] <= key);
                assert_eq!(got, expect, "from={from} key={key}");
            }
        }
    }
}
