//! Key extraction for the merge kernel.
//!
//! The hot loop works on primitive slices only. Every batch is reduced once to
//! a `ts` slice of a native type (`i64` or `i32`) and a group column whose
//! representation depends on the `by` key types:
//!
//! * no `by` keys            -> [`NoGroup`] (every row in group 0)
//! * one Int32-like `by` key -> [`PrimGroup<i32>`]
//! * one other integer-like  -> [`PrimGroup<i64>`] (order-preserving widening)
//! * anything else           -> [`RowGroup`] (arrow `RowConverter`, byte-comparable)
//!
//! For 64-bit (and, for the `i32` path, 32-bit) primitive arrays the key slice
//! borrows the array's value buffer (zero copy). Other types are converted once
//! per batch to an order-preserving `i64` key.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::Hash;

use arrow::array::{Array, ArrayRef};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::{ArrowNativeType, DataType};
use arrow::row::{OwnedRow, RowConverter, Rows, SortField};
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

/// Native key types the kernel is specialised for.
pub trait TsNative:
    ArrowNativeType + Ord + Hash + Eq + Copy + Send + Sync + Debug + 'static
{
    /// Extract a key column of this native type from `arr`.
    fn extract(arr: &ArrayRef) -> Result<KeyCol<Self>>;
    /// Widen to i64 (used for the direct-index group table).
    fn to_i64(self) -> i64;
}

impl TsNative for i64 {
    fn extract(arr: &ArrayRef) -> Result<KeyCol<i64>> {
        use DataType::*;
        Ok(match arr.data_type() {
            Int64 | Timestamp(_, _) | Date64 | Time64(_) | Duration(_) => {
                KeyCol::Borrowed(raw_values::<i64>(arr))
            }
            Int32 | Date32 | Time32(_) => {
                KeyCol::Owned(raw_values::<i32>(arr).iter().map(|v| *v as i64).collect())
            }
            Int16 => KeyCol::Owned(raw_values::<i16>(arr).iter().map(|v| *v as i64).collect()),
            Int8 => KeyCol::Owned(raw_values::<i8>(arr).iter().map(|v| *v as i64).collect()),
            UInt8 => KeyCol::Owned(raw_values::<u8>(arr).iter().map(|v| *v as i64).collect()),
            UInt16 => {
                KeyCol::Owned(raw_values::<u16>(arr).iter().map(|v| *v as i64).collect())
            }
            UInt32 => {
                KeyCol::Owned(raw_values::<u32>(arr).iter().map(|v| *v as i64).collect())
            }
            UInt64 => KeyCol::Owned(
                raw_values::<u64>(arr)
                    .iter()
                    .map(|v| (*v ^ (1u64 << 63)) as i64)
                    .collect(),
            ),
            Float32 => {
                KeyCol::Owned(raw_values::<f32>(arr).iter().map(|v| f64_key(*v as f64)).collect())
            }
            Float64 => KeyCol::Owned(raw_values::<f64>(arr).iter().map(|v| f64_key(*v)).collect()),
            other => return internal_err!("StreamingAsOfJoin: unsupported key type {other}"),
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
            other => internal_err!("StreamingAsOfJoin: i32 key path got type {other}"),
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

/// True for types [`TsNative::extract`] for `i64` supports.
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
    /// Owned group key (for candidates and partition bounds).
    type Owned: Clone + Ord + Hash + Eq + Send + Sync + Debug + 'static;
    /// True if [`GroupKind::pack`] returns packed keys: the kernel then
    /// compares plain i128s (no `by`, or one primitive `by` key).
    const PACKS: bool;

    fn extract(&self, by: &[ArrayRef]) -> Result<Self::Col>;
    /// `(group, ts)` packed into one order-preserving i128 per row, for group
    /// kinds that fit (one primitive key, or none: group 0).
    fn pack<TN: TsNative>(&self, _c: &Self::Col, _ts: &[TN], _n: usize) -> Option<Vec<i128>> {
        None
    }
    fn cmp_at(&self, a: &Self::Col, i: usize, b: &Self::Col, j: usize) -> Ordering;
    fn cmp_owned(&self, a: &Self::Owned, b: &Self::Col, j: usize) -> Ordering;
    fn key_owned(&self, c: &Self::Col, i: usize) -> Self::Owned;
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

/// The group half of a packed key.
#[inline]
pub fn packed_group(k: i128) -> i64 {
    (k >> 64) as i64
}

impl GroupKind for NoGroup {
    type Col = ();
    type Owned = ();
    const PACKS: bool = true;

    fn extract(&self, _by: &[ArrayRef]) -> Result<()> {
        Ok(())
    }
    fn pack<TN: TsNative>(&self, _c: &(), ts: &[TN], n: usize) -> Option<Vec<i128>> {
        Some(ts[..n].iter().map(|t| pack_key(0, TsNative::to_i64(*t))).collect())
    }
    #[inline]
    fn cmp_at(&self, _a: &(), _i: usize, _b: &(), _j: usize) -> Ordering {
        Ordering::Equal
    }
    #[inline]
    fn cmp_owned(&self, _a: &(), _b: &(), _j: usize) -> Ordering {
        Ordering::Equal
    }
    #[inline]
    fn key_owned(&self, _c: &(), _i: usize) {}
}

/// One primitive `by` key.
#[derive(Debug, Default)]
pub struct PrimGroup<GN: TsNative> {
    _marker: std::marker::PhantomData<GN>,
}

impl<GN: TsNative> PrimGroup<GN> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<GN: TsNative> GroupKind for PrimGroup<GN> {
    fn pack<TN: TsNative>(&self, c: &KeyCol<GN>, ts: &[TN], n: usize) -> Option<Vec<i128>> {
        let g = &c.as_slice()[..n];
        Some(
            g.iter()
                .zip(&ts[..n])
                .map(|(g, t)| pack_key(TsNative::to_i64(*g), TsNative::to_i64(*t)))
                .collect(),
        )
    }
    type Col = KeyCol<GN>;
    type Owned = GN;
    const PACKS: bool = true;

    fn extract(&self, by: &[ArrayRef]) -> Result<KeyCol<GN>> {
        GN::extract(&by[0])
    }
    #[inline]
    fn cmp_at(&self, a: &KeyCol<GN>, i: usize, b: &KeyCol<GN>, j: usize) -> Ordering {
        a.as_slice()[i].cmp(&b.as_slice()[j])
    }
    #[inline]
    fn cmp_owned(&self, a: &GN, b: &KeyCol<GN>, j: usize) -> Ordering {
        a.cmp(&b.as_slice()[j])
    }
    #[inline]
    fn key_owned(&self, c: &KeyCol<GN>, i: usize) -> GN {
        c.as_slice()[i]
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
        Ok(Self {
            converter: RowConverter::new(fields)?,
        })
    }

    pub fn supports(types: &[DataType]) -> bool {
        let fields: Vec<SortField> = types.iter().map(|t| SortField::new(t.clone())).collect();
        RowConverter::supports_fields(&fields)
    }
}

impl GroupKind for RowGroup {
    type Col = Rows;
    type Owned = OwnedRow;
    const PACKS: bool = false;

    fn extract(&self, by: &[ArrayRef]) -> Result<Rows> {
        Ok(self.converter.convert_columns(by)?)
    }
    #[inline]
    fn cmp_at(&self, a: &Rows, i: usize, b: &Rows, j: usize) -> Ordering {
        a.row(i).cmp(&b.row(j))
    }
    #[inline]
    fn cmp_owned(&self, a: &OwnedRow, b: &Rows, j: usize) -> Ordering {
        a.row().cmp(&b.row(j))
    }
    fn key_owned(&self, c: &Rows, i: usize) -> OwnedRow {
        c.row(i).owned()
    }
}

/// Merge key `(group, ts)`; without `by` the group is `()`.
pub type Key<TN, G> = (<G as GroupKind>::Owned, TN);

/// Lexicographic `(group, ts)` order.
#[inline]
pub fn cmp_keys<TN: TsNative, G: GroupKind>(a: &Key<TN, G>, b: &Key<TN, G>) -> Ordering {
    a.0.cmp(&b.0).then(a.1.cmp(&b.1))
}

/// Extracted keys of one batch.
pub struct Keys<TN: TsNative, G: GroupKind> {
    pub ts: KeyCol<TN>,
    pub grp: G::Col,
    /// `None` means every row is valid. Only left batches carry nulls; right
    /// batches are filtered first.
    pub valid: Option<arrow::buffer::NullBuffer>,
    pub len: usize,
    /// `GroupKind::pack` of the rows (values of NULL rows are meaningless).
    pub packed: Option<Vec<i128>>,
}

impl<TN: TsNative, G: GroupKind> Keys<TN, G> {
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        self.valid.as_ref().is_none_or(|v| v.is_valid(i))
    }

    pub fn key_at(&self, g: &G, i: usize) -> Key<TN, G> {
        (g.key_owned(&self.grp, i), self.ts.as_slice()[i])
    }

    /// True if the valid rows are sorted by `(group, ts)`. Branch-free on
    /// packed keys without nulls, so it vectorises.
    pub fn is_sorted(&self, g: &G) -> bool {
        let n = self.len;
        match (&self.packed, &self.valid) {
            (Some(p), None) => {
                let mut bad = false;
                for w in p[..n].windows(2) {
                    bad |= w[0] > w[1];
                }
                !bad
            }
            (Some(p), Some(_)) => {
                let mut prev: Option<i128> = None;
                for (i, k) in p[..n].iter().enumerate() {
                    if !self.is_valid(i) {
                        continue;
                    }
                    if prev.is_some_and(|q| q > *k) {
                        return false;
                    }
                    prev = Some(*k);
                }
                true
            }
            (None, _) => {
                let ts = self.ts.as_slice();
                let mut prev: Option<usize> = None;
                for i in (0..n).filter(|&i| self.is_valid(i)) {
                    if let Some(p) = prev {
                        let o = g.cmp_at(&self.grp, p, &self.grp, i).then(ts[p].cmp(&ts[i]));
                        if o == Ordering::Greater {
                            return false;
                        }
                    }
                    prev = Some(i);
                }
                true
            }
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
