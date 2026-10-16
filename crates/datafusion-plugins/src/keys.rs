//! Key extraction for the merge kernel.
//!
//! The hot loop works on primitive slices only. Every batch is reduced once to
//! an `i64` `on` slice (order-preserving: narrower integers, dates and times
//! are widened, unsigned and float keys mapped) and a group column whose
//! representation depends on the `by` key types:
//!
//! * no `by` keys            -> [`NoGroup`] (every row in group 0)
//! * one Int32-like `by` key -> [`PrimGroup<i32>`]
//! * one other integer-like  -> [`PrimGroup<i64>`] (order-preserving widening)
//! * anything else           -> [`RowGroup`] (arrow `RowConverter`, byte-comparable)
//!
//! 64-bit (and, for `PrimGroup<i32>`, 32-bit) primitive arrays are borrowed
//! (zero copy). Other types are converted once per batch.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::Hash;

use arrow::array::{Array, ArrayRef};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::datatypes::{ArrowNativeType, DataType};
use arrow::row::{RowConverter, Rows, SortField};
use datafusion::common::{Result, internal_err};

/// A key column: either a zero-copy view of an Arrow value buffer or an owned
/// converted vector.
pub enum KeyCol<T: ArrowNativeType> {
    Borrowed(ScalarBuffer<T>),
    Owned(Vec<T>),
}

impl<T: ArrowNativeType> KeyCol<T> {
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            KeyCol::Borrowed(b) => b,
            KeyCol::Owned(v) => v,
        }
    }
}

/// Reinterpret the value buffer of a primitive array as `&[T]` (zero copy).
/// The caller guarantees that `T` has the array's primitive width.
fn raw_values<T: ArrowNativeType>(arr: &ArrayRef) -> ScalarBuffer<T> {
    let data = arr.to_data();
    ScalarBuffer::<T>::new(data.buffers()[0].clone(), data.offset(), data.len())
}

/// Order-preserving map from f64 to i64 (the `f64::total_cmp` key), after
/// normalising -0.0 to +0.0 so SQL equality of zeros holds.
#[inline]
fn f64_key(v: f64) -> i64 {
    let v = if v == 0.0 { 0.0 } else { v };
    let b = v.to_bits() as i64;
    b ^ ((((b >> 63) as u64) >> 1) as i64)
}

/// Native key types: `i64` for `on` and wide `by` keys, `i32` for narrow
/// `by` keys.
pub trait TsNative: ArrowNativeType + Ord + Hash + Eq + Copy + Send + Sync + Debug + 'static {
    /// Extract a key column of this native type from `arr`.
    fn extract(arr: &ArrayRef) -> Result<KeyCol<Self>>;
    fn to_i64(self) -> i64;
}

impl TsNative for i64 {
    fn extract(arr: &ArrayRef) -> Result<KeyCol<i64>> {
        use DataType::*;
        Ok(match arr.data_type() {
            Int64 | Timestamp(_, _) | Date64 | Time64(_) | Duration(_) => KeyCol::Borrowed(raw_values::<i64>(arr)),
            Int32 | Date32 | Time32(_) => KeyCol::Owned(raw_values::<i32>(arr).iter().map(|v| *v as i64).collect()),
            Int16 => KeyCol::Owned(raw_values::<i16>(arr).iter().map(|v| *v as i64).collect()),
            Int8 => KeyCol::Owned(raw_values::<i8>(arr).iter().map(|v| *v as i64).collect()),
            UInt8 => KeyCol::Owned(raw_values::<u8>(arr).iter().map(|v| *v as i64).collect()),
            UInt16 => KeyCol::Owned(raw_values::<u16>(arr).iter().map(|v| *v as i64).collect()),
            UInt32 => KeyCol::Owned(raw_values::<u32>(arr).iter().map(|v| *v as i64).collect()),
            UInt64 => KeyCol::Owned(raw_values::<u64>(arr).iter().map(|v| (*v ^ (1u64 << 63)) as i64).collect()),
            Float32 => KeyCol::Owned(raw_values::<f32>(arr).iter().map(|v| f64_key(*v as f64)).collect()),
            Float64 => KeyCol::Owned(raw_values::<f64>(arr).iter().map(|v| f64_key(*v)).collect()),
            other => return internal_err!("AsofExec: unsupported key type {other}"),
        })
    }

    #[inline]
    fn to_i64(self) -> i64 {
        self
    }
}

impl TsNative for i32 {
    fn extract(arr: &ArrayRef) -> Result<KeyCol<i32>> {
        use DataType::*;
        match arr.data_type() {
            Int32 | Date32 | Time32(_) => Ok(KeyCol::Borrowed(raw_values::<i32>(arr))),
            other => internal_err!("AsofExec: the i32 group key path got type {other}"),
        }
    }

    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }
}

/// True for types the i32 kernel handles with a zero-copy view.
pub fn is_i32_like(dt: &DataType) -> bool {
    matches!(dt, DataType::Int32 | DataType::Date32 | DataType::Time32(_))
}

/// True for the `on` types [`TsNative::extract`] for `i64` supports.
pub fn is_supported_ts_type(dt: &DataType) -> bool {
    use DataType::*;
    matches!(
        dt,
        Int8 | Int16
            | Int32
            | Int64
            | UInt8
            | UInt16
            | UInt32
            | UInt64
            | Float32
            | Float64
            | Date32
            | Date64
            | Time32(_)
            | Time64(_)
            | Timestamp(_, _)
            | Duration(_)
    )
}

/// True for single `by` key types the primitive group path handles. Floats are
/// excluded (DataFusion rejects float equality keys for ASOF anyway).
pub fn is_primitive_group_type(dt: &DataType) -> bool {
    is_supported_ts_type(dt) && !matches!(dt, DataType::Float32 | DataType::Float64)
}

/// How rows are grouped (`by` keys) and ordered.
///
/// `cmp` must agree with the order Arrow sorts the key columns in (ascending,
/// non-null values only; null rows never reach these methods).
pub trait GroupKind: Send + Sync + 'static {
    /// Per-batch group column.
    type Col: Send + Sync + 'static;

    fn extract(&self, by: &[ArrayRef]) -> Result<Self::Col>;
    /// `(group, ts)` packed into one order-preserving i128 per row, for group
    /// kinds that fit (one primitive key, or none: group 0).
    fn pack(&self, _c: &Self::Col, _ts: &[i64]) -> Option<Vec<i128>> {
        None
    }
    fn cmp_at(&self, a: &Self::Col, i: usize, b: &Self::Col, j: usize) -> Ordering;
}

/// Ungrouped join: one implicit group.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoGroup;

/// Order-preserving i128 of `(group, ts)`: the signed group in the high 64
/// bits, ts with its sign bit flipped in the low 64 bits.
#[inline]
pub fn pack_key(g: i64, t: i64) -> i128 {
    ((g as i128) << 64) | (((t as u64) ^ (1u64 << 63)) as i128)
}

/// The ts half of a packed key.
#[inline]
pub fn packed_ts(k: i128) -> i64 {
    ((k as u64) ^ (1u64 << 63)) as i64
}

/// The group half of a packed key.
#[inline]
pub fn packed_group(k: i128) -> i64 {
    (k >> 64) as i64
}

impl GroupKind for NoGroup {
    type Col = ();

    fn extract(&self, _by: &[ArrayRef]) -> Result<()> {
        Ok(())
    }
    fn pack(&self, _c: &(), ts: &[i64]) -> Option<Vec<i128>> {
        Some(ts.iter().map(|t| pack_key(0, *t)).collect())
    }
    #[inline]
    fn cmp_at(&self, _a: &(), _i: usize, _b: &(), _j: usize) -> Ordering {
        Ordering::Equal
    }
}

/// One primitive `by` key.
#[derive(Debug, Default)]
pub struct PrimGroup<GN: TsNative> {
    _marker: std::marker::PhantomData<GN>,
}

impl<GN: TsNative> GroupKind for PrimGroup<GN> {
    type Col = KeyCol<GN>;

    fn extract(&self, by: &[ArrayRef]) -> Result<KeyCol<GN>> {
        GN::extract(&by[0])
    }
    fn pack(&self, c: &KeyCol<GN>, ts: &[i64]) -> Option<Vec<i128>> {
        Some(c.as_slice().iter().zip(ts).map(|(g, t)| pack_key(TsNative::to_i64(*g), *t)).collect())
    }
    #[inline]
    fn cmp_at(&self, a: &KeyCol<GN>, i: usize, b: &KeyCol<GN>, j: usize) -> Ordering {
        a.as_slice()[i].cmp(&b.as_slice()[j])
    }
}

/// Multi-column or non-primitive `by` keys, via the Arrow row format.
#[derive(Debug)]
pub struct RowGroup {
    converter: RowConverter,
}

impl RowGroup {
    pub fn try_new(types: &[DataType]) -> Result<Self> {
        let fields = types.iter().map(|t| SortField::new(t.clone())).collect();
        Ok(Self { converter: RowConverter::new(fields)? })
    }

    pub fn supports(types: &[DataType]) -> bool {
        let fields: Vec<SortField> = types.iter().map(|t| SortField::new(t.clone())).collect();
        RowConverter::supports_fields(&fields)
    }
}

impl GroupKind for RowGroup {
    type Col = Rows;

    fn extract(&self, by: &[ArrayRef]) -> Result<Rows> {
        Ok(self.converter.convert_columns(by)?)
    }
    #[inline]
    fn cmp_at(&self, a: &Rows, i: usize, b: &Rows, j: usize) -> Ordering {
        a.row(i).cmp(&b.row(j))
    }
}

/// Extracted keys of one batch.
pub struct Keys<G: GroupKind> {
    pub ts: KeyCol<i64>,
    pub grp: G::Col,
    /// `None` means every row is valid. Only left batches carry nulls; right
    /// batches are filtered first.
    pub valid: Option<NullBuffer>,
    pub len: usize,
    /// `GroupKind::pack` of the rows (values of NULL rows are meaningless).
    pub packed: Option<Vec<i128>>,
}

impl<G: GroupKind> Keys<G> {
    /// Keys of `ts` and `by` (all of length `len`), nulls flagged in `valid`.
    pub fn new(g: &G, ts: &ArrayRef, by: &[ArrayRef], valid: Option<NullBuffer>) -> Result<Self> {
        let (ts, grp) = (i64::extract(ts)?, g.extract(by)?);
        let packed = g.pack(&grp, ts.as_slice());
        Ok(Self { len: ts.as_slice().len(), ts, grp, valid, packed })
    }

    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        self.valid.as_ref().is_none_or(|v| v.is_valid(i))
    }

    /// True if the valid rows are sorted by `(group, ts)`. Branch-free on
    /// packed keys without nulls, so it vectorises.
    pub fn is_sorted(&self, g: &G) -> bool {
        match (&self.packed, &self.valid) {
            (Some(p), None) => {
                let mut bad = false;
                for w in p.windows(2) {
                    bad |= w[0] > w[1];
                }
                !bad
            }
            _ => {
                let mut prev: Option<usize> = None;
                for i in (0..self.len).filter(|&i| self.is_valid(i)) {
                    if prev.is_some_and(|p| self.cmp_at(g, p, i) == Ordering::Greater) {
                        return false;
                    }
                    prev = Some(i);
                }
                true
            }
        }
    }

    /// Compare rows `i` and `j` (both valid) by `(group, ts)`.
    #[inline]
    pub fn cmp_at(&self, g: &G, i: usize, j: usize) -> Ordering {
        match &self.packed {
            Some(p) => p[i].cmp(&p[j]),
            None => g.cmp_at(&self.grp, i, &self.grp, j).then(self.ts.as_slice()[i].cmp(&self.ts.as_slice()[j])),
        }
    }

    /// First / last valid row.
    pub fn first_valid(&self) -> Option<usize> {
        (0..self.len).find(|&i| self.is_valid(i))
    }
    pub fn last_valid(&self) -> Option<usize> {
        (0..self.len).rev().find(|&i| self.is_valid(i))
    }
}
