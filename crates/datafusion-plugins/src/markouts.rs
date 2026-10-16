//! `join_asof_markouts`: backward ASOF joins at several time offsets
//! ("markouts") in one pass over the right input.
//!
//! For each left row and each markout `h` (seconds): the last right row of the
//! same `by` group with `right.on <= left.on + h` (`<` without exact
//! matches), i.e. polars `left.with_columns(on + h).join_asof(right, ...)`
//! for every `h`, side by side. Runs on the shared [`crate::asof`] operator
//! with one cursor per markout over one right buffer that spans
//! `[first + h_min, last + h_max]` of each left batch.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::common::config::ConfigOptions;
use datafusion::common::{Result, internal_datafusion_err, plan_datafusion_err, plan_err};
use datafusion::physical_expr::PhysicalExprRef;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::ExecutionPlan;

use crate::asof::{AsofSpec, build};
use crate::join_asof::two;
use crate::keys::{RowGroup, is_primitive_group_type};
use crate::plan_bytes::PlanNode;

/// A markout: an exact decimal number of seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Markout {
    /// value = mantissa / 10^scale seconds, with no trailing zeros.
    mantissa: i128,
    scale: u32,
}

impl Markout {
    /// Parse a plain decimal (`"60"`, `"-5"`, `"0.5"`).
    pub fn parse(s: &str) -> Result<Self> {
        let t = s.trim();
        let (neg, body) = match t.strip_prefix('-') {
            Some(b) => (true, b),
            None => (false, t.strip_prefix('+').unwrap_or(t)),
        };
        let (int, frac) = body.split_once('.').unwrap_or((body, ""));
        let ok = |p: &str| p.bytes().all(|b| b.is_ascii_digit());
        if (int.is_empty() && frac.is_empty()) || !ok(int) || !ok(frac) || frac.len() > 30 {
            return plan_err!("join_asof_markouts: invalid markout {s:?}");
        }
        let digits = format!("{int}{frac}");
        let mut mantissa: i128 = digits
            .trim_start_matches('0')
            .parse()
            .or_else(|_| if digits.bytes().all(|b| b == b'0') { Ok(0) } else { Err(()) })
            .map_err(|_| plan_datafusion_err!("join_asof_markouts: markout {s:?} is too large"))?;
        let mut scale = frac.len() as u32;
        while scale > 0 && mantissa % 10 == 0 {
            mantissa /= 10;
            scale -= 1;
        }
        if mantissa == 0 {
            scale = 0;
        }
        Ok(Self {
            mantissa: if neg { -mantissa } else { mantissa },
            scale,
        })
    }

    /// The minimal decimal form, for column names: `1`, `0.5`, `-5`.
    pub fn label(&self) -> String {
        let neg = self.mantissa < 0;
        let digits = self.mantissa.unsigned_abs().to_string();
        let s = self.scale as usize;
        let body = if s == 0 {
            digits
        } else if digits.len() > s {
            format!("{}.{}", &digits[..digits.len() - s], &digits[digits.len() - s..])
        } else {
            format!("0.{}{}", "0".repeat(s - digits.len()), digits)
        };
        if neg { format!("-{body}") } else { body }
    }

    /// The markout in units of `10^-digits` seconds, if exact and in range.
    fn in_unit(&self, digits: u32) -> Option<i64> {
        if self.scale > digits {
            return None;
        }
        let v = self.mantissa.checked_mul(10i128.checked_pow(digits - self.scale)?)?;
        i64::try_from(v).ok()
    }
}

impl From<i64> for Markout {
    fn from(v: i64) -> Self {
        Self {
            mantissa: v as i128,
            scale: 0,
        }
    }
}

fn unit_digits(u: &TimeUnit) -> u32 {
    match u {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 3,
        TimeUnit::Microsecond => 6,
        TimeUnit::Nanosecond => 9,
    }
}

/// One output column of the node.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MCol {
    Left(String),
    /// Right column `column` at markout `k` (index into the given list).
    Right { k: usize, column: String },
}

/// Backward ASOF joins of `left.on + h` against `right.on` for every markout
/// `h`, within equal `by` groups. One wide output row per left row, in left
/// order: the left columns, then for each markout (in the given order) the
/// right columns except `on` and `by`, suffixed `_{h}s`.
#[derive(Debug, Clone)]
pub struct JoinAsofMarkouts {
    pub left_on: String,
    pub right_on: String,
    pub by_left: Vec<String>,
    pub by_right: Vec<String>,
    pub markouts: Vec<Markout>,
    pub allow_exact_matches: bool,
    /// Trust the inputs' order instead of requiring it (default: true).
    pub assume_sorted: bool,
    /// With `assume_sorted`: check the order inside every batch.
    pub check_sortedness: bool,
}

impl JoinAsofMarkouts {
    pub fn new(on: impl Into<String>, markouts: Vec<Markout>) -> Self {
        let on = on.into();
        Self {
            left_on: on.clone(),
            right_on: on,
            by_left: Vec::new(),
            by_right: Vec::new(),
            markouts,
            allow_exact_matches: true,
            assume_sorted: true,
            check_sortedness: true,
        }
    }

    pub fn by<I: IntoIterator<Item = S>, S: Into<String>>(mut self, by: I) -> Self {
        self.by_left = by.into_iter().map(Into::into).collect();
        self.by_right = self.by_left.clone();
        self
    }

    pub fn by_left_right<I, J, S, T>(mut self, left: I, right: J) -> Self
    where
        I: IntoIterator<Item = S>,
        J: IntoIterator<Item = T>,
        S: Into<String>,
        T: Into<String>,
    {
        self.by_left = left.into_iter().map(Into::into).collect();
        self.by_right = right.into_iter().map(Into::into).collect();
        self
    }

    pub fn allow_exact_matches(mut self, allow: bool) -> Self {
        self.allow_exact_matches = allow;
        self
    }

    pub fn assume_sorted(mut self, assume: bool) -> Self {
        self.assume_sorted = assume;
        self
    }

    pub fn check_sortedness(mut self, check: bool) -> Self {
        self.check_sortedness = check;
        self
    }

    /// Validate; return the markouts in the `on` column's unit.
    fn shifts(&self, left: &Schema, right: &Schema) -> Result<Vec<i64>> {
        let field = |s: &Schema, n: &str, which: &str| {
            s.field_with_name(n).map(|f| f.data_type().clone()).map_err(|_| {
                plan_datafusion_err!("join_asof_markouts: column '{n}' not found in the {which} input")
            })
        };
        let (lt, rt) = (field(left, &self.left_on, "left")?, field(right, &self.right_on, "right")?);
        let DataType::Timestamp(unit, _) = &lt else {
            return plan_err!("join_asof_markouts: 'on' must be a Timestamp column, got {lt} (left)");
        };
        if lt != rt {
            return plan_err!(
                "join_asof_markouts: 'on' columns must have the same type, got {lt} (left) and {rt} (right)"
            );
        }
        if self.by_left.len() != self.by_right.len() {
            return plan_err!("join_asof_markouts: by_left and by_right must have the same length");
        }
        let mut by_types = Vec::new();
        for (bl, br) in self.by_left.iter().zip(&self.by_right) {
            let (l, r) = (field(left, bl, "left")?, field(right, br, "right")?);
            if l != r {
                return plan_err!(
                    "join_asof_markouts: 'by' columns '{bl}' and '{br}' have different types: {l} and {r}"
                );
            }
            by_types.push(l);
        }
        let ok = match by_types.as_slice() {
            [] => true,
            [t] if is_primitive_group_type(t) => true,
            ts => !ts.iter().any(|t| t.is_floating()) && RowGroup::supports(ts),
        };
        if !ok {
            return plan_err!("join_asof_markouts: unsupported 'by' column types {by_types:?} (float keys are not supported)");
        }
        if self.markouts.is_empty() {
            return plan_err!("join_asof_markouts: give at least one markout");
        }
        let mut seen = HashSet::new();
        let mut shifts = Vec::with_capacity(self.markouts.len());
        for m in &self.markouts {
            if !seen.insert(*m) {
                return plan_err!("join_asof_markouts: duplicate markout {}", m.label());
            }
            let Some(v) = m.in_unit(unit_digits(unit)) else {
                return plan_err!(
                    "join_asof_markouts: markout {}s is not a whole number of {unit:?}s (or is out of range)",
                    m.label()
                );
            };
            shifts.push(v);
        }
        Ok(shifts)
    }

    /// Output columns: left, then per markout the right payload, suffixed.
    fn columns(&self, left: &Schema, right: &Schema) -> Result<Vec<(MCol, String)>> {
        self.shifts(left, right)?;
        let mut out: Vec<(MCol, String)> = left
            .fields()
            .iter()
            .map(|f| (MCol::Left(f.name().clone()), f.name().clone()))
            .collect();
        let mut names: HashSet<String> = out.iter().map(|c| c.1.clone()).collect();
        for (k, m) in self.markouts.iter().enumerate() {
            for f in right.fields() {
                let n = f.name();
                if n == &self.right_on || self.by_right.contains(n) {
                    continue;
                }
                let name = format!("{n}_{}s", m.label());
                if !names.insert(name.clone()) {
                    return plan_err!("join_asof_markouts: output column name '{name}' is not unique");
                }
                out.push((MCol::Right { k, column: n.clone() }, name));
            }
        }
        Ok(out)
    }

    fn output_schema(&self, left: &Schema, right: &Schema) -> Result<SchemaRef> {
        let fields = self
            .columns(left, right)?
            .into_iter()
            .map(|(c, name)| {
                Ok(match c {
                    MCol::Left(n) => left.field_with_name(&n)?.clone(),
                    MCol::Right { column, .. } => {
                        right.field_with_name(&column)?.clone().with_name(name).with_nullable(true)
                    }
                })
            })
            .collect::<Result<Vec<Field>>>()?;
        Ok(Arc::new(Schema::new(fields)))
    }

    /// Build the plan for the output columns `projection` (all if `None`).
    pub fn build(
        &self,
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        projection: Option<&[usize]>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (ls, rs) = (left.schema(), right.schema());
        let shifts = self.shifts(&ls, &rs)?;
        let all = self.columns(&ls, &rs)?;
        let outputs: Vec<(MCol, String)> = match projection {
            Some(p) => p.iter().map(|i| all[*i].clone()).collect(),
            None => all,
        };
        let col = |s: &Schema, n: &str| -> Result<PhysicalExprRef> {
            Ok(Arc::new(Column::new(n, s.index_of(n)?)) as PhysicalExprRef)
        };
        let mut payload_idx: Vec<usize> = Vec::new();
        let mut right_out = Vec::new();
        let mut exec_outputs = Vec::with_capacity(outputs.len());
        for (c, name) in &outputs {
            match c {
                MCol::Left(n) => exec_outputs.push((ls.index_of(n)?, name.clone())),
                MCol::Right { k, column } => {
                    let j = rs.index_of(column)?;
                    let p = payload_idx.iter().position(|x| *x == j).unwrap_or_else(|| {
                        payload_idx.push(j);
                        payload_idx.len() - 1
                    });
                    exec_outputs.push((ls.fields().len() + right_out.len(), name.clone()));
                    right_out.push((*k, p, name.clone()));
                }
            }
        }
        let spec = AsofSpec {
            left_on: col(&ls, &self.left_on)?,
            right_on: col(&rs, &self.right_on)?,
            left_by: self.by_left.iter().map(|b| col(&ls, b)).collect::<Result<_>>()?,
            right_by: self.by_right.iter().map(|b| col(&rs, b)).collect::<Result<_>>()?,
            shifts,
            strict: !self.allow_exact_matches,
            payload_idx,
            right_out,
            assume_sorted: self.assume_sorted,
            check_sortedness: self.check_sortedness,
            nulls_first: (true, true),
        };
        build(left, right, spec, &exec_outputs, config)
    }
}

impl PlanNode for JoinAsofMarkouts {
    fn schema(&self, inputs: &[SchemaRef]) -> Result<SchemaRef> {
        let [l, r] = two(inputs)?;
        self.output_schema(l, r)
    }

    fn required_columns(
        &self,
        inputs: &[SchemaRef],
        projection: &[usize],
    ) -> Result<Vec<Option<Vec<String>>>> {
        let [l, r] = two(inputs)?;
        let cols = self.columns(l, r)?;
        let mut left: Vec<String> = std::iter::once(&self.left_on).chain(&self.by_left).cloned().collect();
        let mut right: Vec<String> =
            std::iter::once(&self.right_on).chain(&self.by_right).cloned().collect();
        for &i in projection {
            match &cols[i].0 {
                MCol::Left(n) => left.push(n.clone()),
                MCol::Right { column, .. } => right.push(column.clone()),
            }
        }
        let keep = |s: &Schema, want: &[String]| -> Vec<String> {
            s.fields().iter().map(|f| f.name().clone()).filter(|n| want.contains(n)).collect()
        };
        Ok(vec![Some(keep(l, &left)), Some(keep(r, &right))])
    }

    fn build(
        &self,
        schemas: &[SchemaRef],
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        projection: &[usize],
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Output names depend on the full schemas; the pruned inputs keep
        // every column the projection needs, under the same names.
        let [l, r] = two(schemas)?;
        let all = self.columns(l, r)?;
        let [left, right]: [Arc<dyn ExecutionPlan>; 2] = inputs
            .try_into()
            .map_err(|_| plan_datafusion_err!("join_asof_markouts takes two inputs"))?;
        let pruned = self.columns(&left.schema(), &right.schema())?;
        let proj = projection
            .iter()
            .map(|&i| {
                pruned.iter().position(|c| c.1 == all[i].1).ok_or_else(|| {
                    internal_datafusion_err!("join_asof_markouts: column {} missing", all[i].1)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.build(left, right, Some(&proj), config)
    }
}

