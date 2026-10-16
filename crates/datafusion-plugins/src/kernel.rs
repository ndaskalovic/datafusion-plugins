//! Batch-level building blocks: galloping search, right-row references, key
//! extraction and output gathering.

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch, UInt32Array, new_null_array};
use arrow::buffer::NullBuffer;
use arrow::compute::{filter, filter_record_batch, interleave, take};
use arrow::datatypes::DataType;
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

/// Extract keys of a right batch and keep the payload columns. Rows with a
/// NULL ts or `by` key can never match, so they are filtered out here.
/// Returns `None` for a batch with no usable rows.
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

/// Column `j` of `batches` gathered by `refs` (`NONE` = null).
pub fn take_refs(batches: &[&RecordBatch], j: usize, dt: &DataType, refs: &[u64]) -> Result<ArrayRef> {
    let mut src: Option<usize> = None;
    let mut single = true;
    for r in refs.iter().filter(|r| **r != NONE) {
        match src {
            None => src = Some(ref_src(*r)),
            Some(s) if s != ref_src(*r) => {
                single = false;
                break;
            }
            Some(_) => {}
        }
    }
    match (src, single) {
        (None, _) => Ok(new_null_array(dt, refs.len())),
        (Some(s), true) => {
            let idx: UInt32Array = refs
                .iter()
                .map(|r| (*r != NONE).then(|| ref_row(*r) as u32))
                .collect();
            Ok(take(batches[s].column(j).as_ref(), &idx, None)?)
        }
        _ => {
            let null = new_null_array(dt, 1);
            let mut arrs: Vec<&dyn Array> = vec![null.as_ref()];
            arrs.extend(batches.iter().map(|b| b.column(j).as_ref()));
            let idx: Vec<(usize, usize)> = refs
                .iter()
                .map(|r| if *r == NONE { (0, 0) } else { (ref_src(*r) + 1, ref_row(*r)) })
                .collect();
            Ok(interleave(&arrs, &idx)?)
        }
    }
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
