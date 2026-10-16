//! The `join_asof` node: a backward ASOF join with polars
//! `join_asof(strategy="backward")` semantics, optionally at several time
//! offsets ("markouts") in one pass over the right input.
//!
//! With `by`, each input must arrive sorted by `(by..., on)`; without, by
//! `on`. Unless `assume_sorted`, the node requires that order and DataFusion
//! adds a sort only where it can't prove it; with `assume_sorted` the order
//! is trusted and checked while streaming (see [`crate::asof`]).

use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::common::config::ConfigOptions;
use datafusion::common::{Result, plan_datafusion_err, plan_err};
use datafusion::physical_expr::PhysicalExprRef;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::ExecutionPlan;

use crate::PlanNode;
use crate::asof::{AsofSpec, build};
use crate::keys::{RowGroup, is_primitive_group_type, is_supported_ts_type};

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
        Ok(Self { mantissa: if neg { -mantissa } else { mantissa }, scale })
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
        Self { mantissa: v as i128, scale: 0 }
    }
}

/// Which input an output column comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// One output column: `column` of `side`, named `name`; right columns at
/// markout `offset` (0 without markouts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputColumn {
    pub side: Side,
    pub column: String,
    pub name: String,
    pub offset: usize,
}

/// A backward ASOF join, as polars `join_asof(strategy="backward")`: for
/// each left row, the last right row with equal `by` values and
/// `right_on <= left_on` (`<` without `allow_exact_matches`). One output row
/// per left row, in left order.
///
/// With `markouts` (seconds; `on` must be a timestamp), one such join per
/// markout `h` of `left_on + h`, side by side: the left columns, then for
/// each markout the right columns, suffixed `_{h}s`.
#[derive(Debug, Clone)]
pub struct JoinAsof {
    pub left_on: String,
    pub right_on: String,
    pub by_left: Vec<String>,
    pub by_right: Vec<String>,
    pub allow_exact_matches: bool,
    /// Without markouts: appended to right names that clash with left ones.
    pub suffix: String,
    pub markouts: Option<Vec<Markout>>,
    /// Trust the inputs' order instead of requiring it.
    pub assume_sorted: bool,
    /// With `assume_sorted`: check the order inside every batch.
    pub check_sortedness: bool,
}

impl JoinAsof {
    /// Join on column `on` of both inputs.
    pub fn new(on: impl Into<String>) -> Self {
        let on = on.into();
        Self::new_with(on.clone(), on)
    }

    /// Join on `left_on` of the left input and `right_on` of the right.
    pub fn new_with(left_on: impl Into<String>, right_on: impl Into<String>) -> Self {
        Self {
            left_on: left_on.into(),
            right_on: right_on.into(),
            by_left: Vec::new(),
            by_right: Vec::new(),
            allow_exact_matches: true,
            suffix: "_right".to_string(),
            markouts: None,
            assume_sorted: false,
            check_sortedness: true,
        }
    }

    /// Equality columns, the same names on both sides.
    pub fn by<I: IntoIterator<Item = S>, S: Into<String>>(mut self, by: I) -> Self {
        self.by_left = by.into_iter().map(Into::into).collect();
        self.by_right = self.by_left.clone();
        self
    }

    /// Equality columns, named per side.
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

    pub fn suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }

    /// Join at each of `markouts` (see the type docs).
    pub fn markouts(mut self, markouts: Vec<Markout>) -> Self {
        self.markouts = Some(markouts);
        self
    }

    pub fn check_sortedness(mut self, check: bool) -> Self {
        self.check_sortedness = check;
        self
    }

    pub fn assume_sorted(mut self, assume: bool) -> Self {
        self.assume_sorted = assume;
        self
    }

    fn name(&self) -> &'static str {
        if self.markouts.is_some() { "join_asof_markouts" } else { "join_asof" }
    }

    /// Check the arguments against the input schemas; the offsets, in the
    /// `on` column's units.
    pub fn validate(&self, left: &Schema, right: &Schema) -> Result<Vec<i64>> {
        let node = self.name();
        if self.by_left.len() != self.by_right.len() {
            return plan_err!("{node}: by_left and by_right must have the same length");
        }
        let find = |s: &Schema, name: &str, which: &str| -> Result<DataType> {
            s.field_with_name(name).map(|f| f.data_type().clone()).map_err(|_| {
                plan_datafusion_err!(
                    "{node}: column '{name}' not found in the {which} input; columns: [{}]",
                    s.fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>().join(", ")
                )
            })
        };
        for (s, which) in [(left, "left"), (right, "right")] {
            let mut seen = HashSet::new();
            if let Some(f) = s.fields().iter().find(|f| !seen.insert(f.name())) {
                return plan_err!("{node}: the {which} input has two columns named '{}'; rename one first", f.name());
            }
        }
        let lt = find(left, &self.left_on, "left")?;
        let rt = find(right, &self.right_on, "right")?;
        if lt != rt {
            return plan_err!("{node}: 'on' columns must have the same type, got {lt} (left) and {rt} (right)");
        }
        let mut by_types = Vec::new();
        for (bl, br) in self.by_left.iter().zip(&self.by_right) {
            let (l, r) = (find(left, bl, "left")?, find(right, br, "right")?);
            if l != r {
                return plan_err!("{node}: 'by' columns '{bl}' and '{br}' have different types: {l} and {r}");
            }
            by_types.push(l);
        }
        let ok = match by_types.as_slice() {
            [] => true,
            [t] if is_primitive_group_type(t) => true,
            ts => !ts.iter().any(|t| t.is_floating()) && RowGroup::supports(ts),
        };
        if !ok {
            return plan_err!("{node}: unsupported 'by' column types {by_types:?} (float keys are not supported)");
        }
        let Some(markouts) = &self.markouts else {
            if !is_supported_ts_type(&lt) {
                return plan_err!(
                    "{node}: unsupported 'on' type {lt} (integers, floats, dates, times, timestamps and \
                     durations are supported)"
                );
            }
            return Ok(vec![0]);
        };
        let DataType::Timestamp(unit, _) = &lt else {
            return plan_err!("{node}: 'on' must be a Timestamp column, got {lt} (left)");
        };
        let digits = match unit {
            TimeUnit::Second => 0,
            TimeUnit::Millisecond => 3,
            TimeUnit::Microsecond => 6,
            TimeUnit::Nanosecond => 9,
        };
        if markouts.is_empty() {
            return plan_err!("{node}: give at least one markout");
        }
        let mut seen = HashSet::new();
        markouts
            .iter()
            .map(|m| {
                if !seen.insert(*m) {
                    return plan_err!("{node}: duplicate markout {}", m.label());
                }
                m.in_unit(digits).ok_or_else(|| {
                    plan_datafusion_err!(
                        "{node}: markout {}s is not a whole number of {unit:?}s (or is out of range)",
                        m.label()
                    )
                })
            })
            .collect()
    }

    /// The output columns: every left column, then the right columns except
    /// the `by` keys and the `on` column when it has the left `on` name.
    /// Without markouts (polars), a right name that is already taken gets
    /// `suffix`, and it is an error if that is taken too; with markouts the
    /// right columns repeat per markout `h`, suffixed `_{h}s`.
    pub fn output_columns(&self, left: &Schema, right: &Schema) -> Result<Vec<OutputColumn>> {
        self.validate(left, right)?;
        let col = |side, column: &String, name, offset| OutputColumn { side, column: column.clone(), name, offset };
        let mut out: Vec<OutputColumn> = left.fields().iter().map(|f| col(Side::Left, f.name(), f.name().clone(), 0)).collect();
        let right_cols: Vec<&String> = right
            .fields()
            .iter()
            .map(|f| f.name())
            .filter(|n| !(self.by_right.contains(n) || (*n == &self.right_on && self.right_on == self.left_on)))
            .collect();
        let mut names: HashSet<String> = out.iter().map(|c| c.name.clone()).collect();
        let labels = match &self.markouts {
            Some(ms) => ms.iter().map(|m| Some(m.label())).collect(),
            None => vec![None],
        };
        for (k, label) in labels.iter().enumerate() {
            for n in &right_cols {
                let name = match label {
                    Some(l) => format!("{n}_{l}s"),
                    None if names.contains(*n) => format!("{n}{}", self.suffix),
                    None => n.to_string(),
                };
                if !names.insert(name.clone()) {
                    return plan_err!(
                        "{}: output column name '{name}' already exists; rename the column{}",
                        self.name(),
                        if label.is_some() { "" } else { " or use a different suffix" }
                    );
                }
                out.push(col(Side::Right, n, name, k));
            }
        }
        Ok(out)
    }

    /// Schema of `outputs`: right columns are nullable.
    fn output_schema(outputs: &[OutputColumn], left: &Schema, right: &Schema) -> Result<SchemaRef> {
        let fields = outputs
            .iter()
            .map(|o| {
                Ok(match o.side {
                    Side::Left => left.field_with_name(&o.column)?.clone().with_name(o.name.clone()),
                    Side::Right => right.field_with_name(&o.column)?.clone().with_name(o.name.clone()).with_nullable(true),
                })
            })
            .collect::<Result<Vec<Field>>>()?;
        Ok(Arc::new(Schema::new(fields)))
    }

    /// Build the join with all of [`Self::output_columns`].
    pub fn build(
        &self,
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let outputs = self.output_columns(&left.schema(), &right.schema())?;
        self.build_outputs(left, right, &outputs, config)
    }

    /// Build the join producing `outputs` (columns resolved by name).
    fn build_outputs(
        &self,
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        outputs: &[OutputColumn],
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (ls, rs) = (left.schema(), right.schema());
        let shifts = self.validate(&ls, &rs)?;
        let col = |s: &Schema, name: &str| -> Result<PhysicalExprRef> { Ok(Arc::new(Column::new(name, s.index_of(name)?))) };
        let mut payload_idx: Vec<usize> = Vec::new();
        let mut right_out = Vec::new();
        let mut exec_outputs = Vec::with_capacity(outputs.len());
        for o in outputs {
            match o.side {
                Side::Left => exec_outputs.push((ls.index_of(&o.column)?, o.name.clone())),
                Side::Right => {
                    let j = rs.index_of(&o.column)?;
                    let p = payload_idx.iter().position(|x| *x == j).unwrap_or_else(|| {
                        payload_idx.push(j);
                        payload_idx.len() - 1
                    });
                    exec_outputs.push((ls.fields().len() + right_out.len(), o.name.clone()));
                    right_out.push((o.offset, p, o.name.clone()));
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
            nulls_first: (None, None),
        };
        build(left, right, spec, &exec_outputs, config)
    }
}

impl PlanNode for JoinAsof {
    fn schema(&self, inputs: &[SchemaRef]) -> Result<SchemaRef> {
        let [l, r] = two(inputs)?;
        Self::output_schema(&self.output_columns(l, r)?, l, r)
    }

    fn required_columns(&self, inputs: &[SchemaRef], projection: &[usize]) -> Result<Vec<Option<Vec<String>>>> {
        let [l, r] = two(inputs)?;
        let outputs = self.output_columns(l, r)?;
        let keep = |schema: &Schema, side: Side, keys: Vec<&String>| -> Vec<String> {
            schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|n| {
                    keys.contains(&n) || projection.iter().any(|&i| outputs[i].side == side && &outputs[i].column == n)
                })
                .collect()
        };
        let lk = std::iter::once(&self.left_on).chain(&self.by_left).collect();
        let rk = std::iter::once(&self.right_on).chain(&self.by_right).collect();
        Ok(vec![Some(keep(l, Side::Left, lk)), Some(keep(r, Side::Right, rk))])
    }

    fn build(
        &self,
        schemas: &[SchemaRef],
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        projection: &[usize],
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Output names come from the full schemas; the pruned inputs keep
        // every column the projection needs, under the same names.
        let [l, r] = two(schemas)?;
        let all = self.output_columns(l, r)?;
        let outputs: Vec<OutputColumn> = projection.iter().map(|i| all[*i].clone()).collect();
        let [left, right]: [Arc<dyn ExecutionPlan>; 2] =
            inputs.try_into().map_err(|_| plan_datafusion_err!("{} takes two inputs", self.name()))?;
        self.build_outputs(left, right, &outputs, config)
    }
}

/// The two schemas of a two-input node.
fn two(inputs: &[SchemaRef]) -> Result<[&Schema; 2]> {
    match inputs {
        [l, r] => Ok([l.as_ref(), r.as_ref()]),
        _ => plan_err!("expected two inputs, got {}", inputs.len()),
    }
}
