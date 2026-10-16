//! Building a `join_asof` plan from two physical inputs, with polars
//! `join_asof(strategy="backward")` semantics.
//!
//! With `by`, each input must arrive sorted by `(by..., on)`; without, by
//! `on`. Unless `assume_sorted`, the node requires that order and DataFusion
//! adds a sort only where it can't prove it; with `assume_sorted` the order
//! is trusted and checked while streaming (see [`crate::asof`]).

use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::config::ConfigOptions;
use datafusion::common::{Result, plan_datafusion_err, plan_err};
use datafusion::physical_expr::PhysicalExprRef;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::ExecutionPlan;

use crate::asof::{AsofSpec, build};
use crate::keys::{RowGroup, is_primitive_group_type, is_supported_ts_type};
use crate::plan_bytes::PlanNode;

/// Which input an output column comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// One output column: `column` of `side`, named `name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputColumn {
    pub side: Side,
    pub column: String,
    pub name: String,
}

/// A backward ASOF join, as polars `join_asof(strategy="backward")`: for
/// each left row, the last right row with equal `by` values and
/// `right_on <= left_on` (`<` without `allow_exact_matches`). One output row
/// per left row, in left order.
#[derive(Debug, Clone)]
pub struct JoinAsof {
    pub left_on: String,
    pub right_on: String,
    pub by_left: Vec<String>,
    pub by_right: Vec<String>,
    pub allow_exact_matches: bool,
    pub suffix: String,
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

    pub fn check_sortedness(mut self, check: bool) -> Self {
        self.check_sortedness = check;
        self
    }

    pub fn assume_sorted(mut self, assume: bool) -> Self {
        self.assume_sorted = assume;
        self
    }

    /// Check the key columns of both schemas: present, same types, supported.
    pub fn validate(&self, left: &Schema, right: &Schema) -> Result<()> {
        if self.by_left.len() != self.by_right.len() {
            return plan_err!("join_asof: by_left and by_right must have the same length");
        }
        let find = |s: &Schema, name: &str, which: &str| -> Result<DataType> {
            s.field_with_name(name).map(|f| f.data_type().clone()).map_err(|_| {
                plan_datafusion_err!(
                    "join_asof: column '{name}' not found in the {which} input; columns: [{}]",
                    s.fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>().join(", ")
                )
            })
        };
        for (s, which) in [(left, "left"), (right, "right")] {
            let mut seen = HashSet::new();
            for f in s.fields() {
                if !seen.insert(f.name()) {
                    return plan_err!(
                        "join_asof: the {which} input has two columns named '{}'; \
                         rename one first",
                        f.name()
                    );
                }
            }
        }
        let lt = find(left, &self.left_on, "left")?;
        let rt = find(right, &self.right_on, "right")?;
        if lt != rt {
            return plan_err!(
                "join_asof: 'on' columns have different types: {lt} (left) and {rt} (right)"
            );
        }
        if !is_supported_ts_type(&lt) {
            return plan_err!(
                "join_asof: unsupported 'on' type {lt} (integers, floats, dates, times, \
                 timestamps and durations are supported)"
            );
        }
        let mut by_types = Vec::new();
        for (bl, br) in self.by_left.iter().zip(&self.by_right) {
            let (l, r) = (find(left, bl, "left")?, find(right, br, "right")?);
            if l != r {
                return plan_err!(
                    "join_asof: 'by' columns '{bl}' and '{br}' have different types: {l} and {r}"
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
            return plan_err!(
                "join_asof: unsupported 'by' column types {by_types:?} (float keys are not supported)"
            );
        }
        Ok(())
    }

    /// polars' output: every left column, then every right column in order,
    /// except the `by` keys and the `on` column when it has the left `on`
    /// name. A right name that is already taken gets `suffix`; if that is
    /// taken too, it is an error (as in polars).
    pub fn output_columns(&self, left: &Schema, right: &Schema) -> Result<Vec<OutputColumn>> {
        let mut out: Vec<OutputColumn> = left
            .fields()
            .iter()
            .map(|f| OutputColumn {
                side: Side::Left,
                column: f.name().clone(),
                name: f.name().clone(),
            })
            .collect();
        for f in right.fields() {
            let n = f.name();
            if self.by_right.contains(n) || (n == &self.right_on && self.right_on == self.left_on) {
                continue;
            }
            let mut name = n.clone();
            if out.iter().any(|c| c.name == name) {
                name = format!("{n}{}", self.suffix);
                if out.iter().any(|c| c.name == name) {
                    return plan_err!(
                        "join_asof: column with name '{name}' already exists; rename the \
                         column or use a different suffix"
                    );
                }
            }
            out.push(OutputColumn {
                side: Side::Right,
                column: n.clone(),
                name,
            });
        }
        Ok(out)
    }

    /// Schema of [`Self::output_columns`]: right columns are nullable.
    pub fn output_schema(&self, left: &Schema, right: &Schema) -> Result<SchemaRef> {
        output_schema(&self.output_columns(left, right)?, left, right)
    }

    /// Build the join with polars' output columns.
    pub fn build(
        &self,
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let outputs = self.output_columns(&left.schema(), &right.schema())?;
        self.build_with_outputs(left, right, &outputs, config)
    }

    /// Build the join with an explicit output column list.
    pub fn build_with_outputs(
        &self,
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        outputs: &[OutputColumn],
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (ls, rs) = (left.schema(), right.schema());
        self.validate(&ls, &rs)?;
        let col = |s: &Schema, name: &str| -> Result<PhysicalExprRef> {
            Ok(Arc::new(Column::new(name, s.index_of(name)?)) as PhysicalExprRef)
        };
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
                    right_out.push((0, p, o.name.clone()));
                }
            }
        }
        let spec = AsofSpec {
            left_on: col(&ls, &self.left_on)?,
            right_on: col(&rs, &self.right_on)?,
            left_by: self.by_left.iter().map(|b| col(&ls, b)).collect::<Result<_>>()?,
            right_by: self.by_right.iter().map(|b| col(&rs, b)).collect::<Result<_>>()?,
            shifts: vec![0],
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

/// Arrow schema of `outputs` over the two input schemas.
pub fn output_schema(outputs: &[OutputColumn], left: &Schema, right: &Schema) -> Result<SchemaRef> {
    let fields = outputs
        .iter()
        .map(|o| {
            Ok(match o.side {
                Side::Left => left.field_with_name(&o.column)?.clone().with_name(o.name.clone()),
                Side::Right => right
                    .field_with_name(&o.column)?
                    .clone()
                    .with_name(o.name.clone())
                    .with_nullable(true),
            })
        })
        .collect::<Result<Vec<Field>>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

impl PlanNode for JoinAsof {
    fn schema(&self, inputs: &[SchemaRef]) -> Result<SchemaRef> {
        let [l, r] = two(inputs)?;
        self.validate(l, r)?;
        self.output_schema(l, r)
    }

    fn required_columns(
        &self,
        inputs: &[SchemaRef],
        projection: &[usize],
    ) -> Result<Vec<Option<Vec<String>>>> {
        let [l, r] = two(inputs)?;
        let outputs = self.output_columns(l, r)?;
        let keep = |schema: &Schema, side: Side, keys: Vec<&String>| -> Vec<String> {
            schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|n| {
                    keys.contains(&n)
                        || projection.iter().any(|&i| outputs[i].side == side && &outputs[i].column == n)
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
        let [l, r] = two(schemas)?;
        let all = self.output_columns(l, r)?;
        let outputs: Vec<OutputColumn> = projection.iter().map(|i| all[*i].clone()).collect();
        let [left, right]: [Arc<dyn ExecutionPlan>; 2] = inputs
            .try_into()
            .map_err(|_| plan_datafusion_err!("join_asof takes two inputs"))?;
        self.build_with_outputs(left, right, &outputs, config)
    }
}

/// The two schemas of a two-input node.
pub(crate) fn two(inputs: &[SchemaRef]) -> Result<[&Schema; 2]> {
    match inputs {
        [l, r] => Ok([l.as_ref(), r.as_ref()]),
        _ => plan_err!("expected two inputs, got {}", inputs.len()),
    }
}
