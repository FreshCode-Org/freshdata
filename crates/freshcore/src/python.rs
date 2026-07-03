use std::collections::HashSet;
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::arrays::{Column, ColumnData, Frame};
use crate::kernels::{casts, duplicates, missing, outliers, profile, strings};
use crate::plan::{CleanPlan, StringCase};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(execute_plan, m)?)?;
    Ok(())
}

#[pyfunction]
fn execute_plan(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<PyObject> {
    let mut timings = Vec::new();
    let mut frame = frame_from_payload(payload)?;
    let plan = plan_from_payload(payload)?;
    let rows_before = frame.nrows;
    let cols_before = frame.columns.len();
    let missing_before = frame.null_cells();
    let mut actions = Vec::new();
    let mut columns_dropped = Vec::new();
    let mut columns_imputed = Vec::new();
    let mut duplicates_removed = 0usize;
    let mut outliers_handled = 0usize;

    let t = Instant::now();
    apply_renames(&mut frame, &plan, &mut actions);
    timings.push(("column_names".to_string(), t.elapsed().as_secs_f64()));

    let t = Instant::now();
    for action in strings::clean_strings(&mut frame, &plan) {
        if action.trimmed > 0 {
            actions.push(action_dict(
                "strip_whitespace",
                Some(action.column.clone()),
                "trimmed surrounding whitespace",
                action.trimmed,
            ));
        }
        if action.case_normalized > 0 {
            actions.push(action_dict(
                "normalize_case",
                Some(action.column.clone()),
                "normalized string case",
                action.case_normalized,
            ));
        }
        if action.sentinels > 0 {
            actions.push(action_dict(
                "normalize_sentinels",
                Some(action.column),
                "replaced sentinel strings (\"N/A\", \"-\", \"\", …) with missing",
                action.sentinels,
            ));
        }
    }
    timings.push(("clean_strings".to_string(), t.elapsed().as_secs_f64()));

    if plan.drop_empty_columns {
        let t = Instant::now();
        let dropped = missing::drop_empty_columns(&mut frame);
        if !dropped.is_empty() {
            columns_dropped.extend(dropped.clone());
            actions.push(action_dict(
                "drop_empty_columns",
                None,
                &format!("dropped {} all-missing column(s): {}", dropped.len(), dropped.join(", ")),
                dropped.len(),
            ));
        }
        timings.push(("drop_empty_columns".to_string(), t.elapsed().as_secs_f64()));
    }

    if plan.drop_empty_rows {
        let t = Instant::now();
        let dropped = missing::drop_empty_rows(&mut frame);
        if dropped > 0 {
            actions.push(action_dict(
                "drop_empty_rows",
                None,
                &format!("dropped {dropped} all-missing row(s)"),
                dropped,
            ));
        }
        timings.push(("drop_empty_rows".to_string(), t.elapsed().as_secs_f64()));
    }

    let t = Instant::now();
    for action in casts::infer_and_cast(&mut frame, &plan) {
        let suffix = if action.coerced > 0 {
            format!(" ({} unparseable value(s) set to missing)", action.coerced)
        } else {
            String::new()
        };
        actions.push(action_dict(
            "fix_dtypes",
            Some(action.column),
            &format!("converted to {}{}", action.target, suffix),
            action.count,
        ));
    }
    timings.push(("fix_dtypes".to_string(), t.elapsed().as_secs_f64()));

    if plan.drop_duplicates {
        let t = Instant::now();
        let before = frame.nrows;
        let dropped = duplicates::drop_duplicates(&mut frame, &plan.duplicate_keep);
        duplicates_removed = dropped;
        if dropped > 0 {
            let pct = 100.0 * dropped as f64 / before as f64;
            actions.push(action_dict(
                "drop_duplicates",
                None,
                &format!(
                    "dropped {dropped} duplicate row(s) ({pct:.1}% of rows, keep={:?})",
                    plan.duplicate_keep
                ),
                dropped,
            ));
        }
        timings.push(("drop_duplicates".to_string(), t.elapsed().as_secs_f64()));
    }

    let t = Instant::now();
    for action in missing::impute(&mut frame, plan.impute.as_deref()) {
        columns_imputed.push(action.column.clone());
        actions.push(action_dict(
            "impute",
            Some(action.column),
            &format!(
                "filled {} missing value(s) with {} ({})",
                action.count, action.strategy, action.value
            ),
            action.count,
        ));
    }
    timings.push(("impute".to_string(), t.elapsed().as_secs_f64()));

    let t = Instant::now();
    for action in outliers::handle_outliers(
        &mut frame,
        plan.outliers.as_deref(),
        &plan.outlier_method,
        plan.outlier_factor,
    ) {
        outliers_handled += action.count;
        let description = if action.action == "clip" {
            format!(
                "clipped {} outlier(s) to [{:.6}, {:.6}] ({}, factor {:.6})",
                action.count, action.lower, action.upper, plan.outlier_method, plan.outlier_factor
            )
        } else {
            format!(
                "flagged {} outlier(s) in new column {:?} ({}, factor {:.6})",
                action.count,
                action.flag_column.unwrap_or_default(),
                plan.outlier_method,
                plan.outlier_factor
            )
        };
        actions.push(action_dict(
            "outliers",
            Some(action.column),
            &description,
            action.count,
        ));
    }
    timings.push(("outliers".to_string(), t.elapsed().as_secs_f64()));

    let profiles = profile::profile(&frame);
    let result = PyDict::new_bound(py);
    result.set_item("rows_before", rows_before)?;
    result.set_item("rows_after", frame.nrows)?;
    result.set_item("cols_before", cols_before)?;
    result.set_item("cols_after", frame.columns.len())?;
    result.set_item("missing_before", missing_before)?;
    result.set_item("missing_after", frame.null_cells())?;
    result.set_item("duplicates_removed", duplicates_removed)?;
    result.set_item("outliers_handled", outliers_handled)?;
    result.set_item("columns_dropped", columns_dropped)?;
    result.set_item("columns_imputed", columns_imputed)?;
    result.set_item("actions", actions)?;
    result.set_item("stage_timings", timings)?;
    result.set_item("profile", profiles_to_py(py, &profiles)?)?;
    result.set_item("columns", columns_to_py(py, &frame.columns)?)?;
    Ok(result.into())
}

fn frame_from_payload(payload: &Bound<'_, PyDict>) -> PyResult<Frame> {
    let columns_obj = payload
        .get_item("columns")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing columns"))?;
    let columns_list = columns_obj.downcast::<PyList>()?;
    let mut columns = Vec::new();
    for item in columns_list {
        let col = item.downcast::<PyDict>()?;
        let name: String = col.get_item("name")?.unwrap().extract()?;
        let dtype: String = col.get_item("dtype")?.unwrap().extract()?;
        let values_obj = col.get_item("values")?.unwrap();
        let values = values_obj.downcast::<PyList>()?;
        let data = match dtype.as_str() {
            "float" => ColumnData::Float(
                values
                    .iter()
                    .map(|v| {
                        if v.is_none() {
                            Ok(None)
                        } else {
                            Ok(Some(v.extract::<f64>()?))
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            "bool" => ColumnData::Bool(
                values
                    .iter()
                    .map(|v| {
                        if v.is_none() {
                            Ok(None)
                        } else {
                            Ok(Some(v.extract::<bool>()?))
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            _ => ColumnData::Utf8(
                values
                    .iter()
                    .map(|v| {
                        if v.is_none() {
                            Ok(None)
                        } else {
                            Ok(Some(v.extract::<String>()?))
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            ),
        };
        columns.push(Column { name, data });
    }
    Frame::new(columns).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

fn plan_from_payload(payload: &Bound<'_, PyDict>) -> PyResult<CleanPlan> {
    let cfg_obj = payload
        .get_item("config")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("missing config"))?;
    let cfg = cfg_obj.downcast::<PyDict>()?;
    let mut plan = CleanPlan::default();
    plan.rename_map = extract_pairs(cfg, "rename_map")?;
    plan.strip_whitespace = get_bool(cfg, "strip_whitespace", true)?;
    plan.normalize_sentinels = get_bool(cfg, "normalize_sentinels", true)?;
    plan.sentinels = extract_strings(cfg, "sentinels")?.into_iter().collect::<HashSet<_>>();
    plan.string_case = match get_optional_string(cfg, "string_case")?.as_deref() {
        Some("lower") => StringCase::Lower,
        Some("upper") => StringCase::Upper,
        _ => StringCase::Preserve,
    };
    plan.drop_empty_columns = get_bool(cfg, "drop_empty_columns", true)?;
    plan.drop_empty_rows = get_bool(cfg, "drop_empty_rows", true)?;
    plan.drop_duplicates = get_bool(cfg, "drop_duplicates", true)?;
    plan.duplicate_keep = get_string(cfg, "duplicate_keep", "first")?;
    plan.fix_dtypes = get_bool(cfg, "fix_dtypes", true)?;
    plan.numeric_threshold = get_float(cfg, "numeric_threshold", 0.95)?;
    plan.preserve_leading_zeros = get_bool(cfg, "preserve_leading_zeros", true)?;
    plan.impute = get_optional_string(cfg, "impute")?;
    plan.outliers = get_optional_string(cfg, "outliers")?;
    plan.outlier_method = get_string(cfg, "outlier_method", "iqr")?;
    plan.outlier_factor = get_float(cfg, "outlier_factor", 1.5)?;
    Ok(plan)
}

fn apply_renames(frame: &mut Frame, plan: &CleanPlan, actions: &mut Vec<PyObject>) {
    if plan.rename_map.is_empty() {
        return;
    }
    let mut count = 0usize;
    for column in &mut frame.columns {
        if let Some((_, new)) = plan.rename_map.iter().find(|(old, _)| old == &column.name) {
            column.name = new.clone();
            count += 1;
        }
    }
    if count > 0 {
        Python::with_gil(|py| {
            actions.push(action_dict(
                "column_names",
                None,
                &format!("renamed {count} column(s)"),
                count,
            ).into_py(py));
        });
    }
}

fn action_dict(step: &str, column: Option<String>, description: &str, count: usize) -> PyObject {
    Python::with_gil(|py| {
        let d = PyDict::new_bound(py);
        d.set_item("step", step).unwrap();
        d.set_item("column", column).unwrap();
        d.set_item("description", description).unwrap();
        d.set_item("count", count).unwrap();
        d.into()
    })
}

fn columns_to_py(py: Python<'_>, columns: &[Column]) -> PyResult<PyObject> {
    let out = PyList::empty_bound(py);
    for column in columns {
        let d = PyDict::new_bound(py);
        d.set_item("name", &column.name)?;
        match &column.data {
            ColumnData::Float(values) => {
                d.set_item("dtype", "float")?;
                d.set_item("values", values)?;
            }
            ColumnData::Bool(values) => {
                d.set_item("dtype", "bool")?;
                d.set_item("values", values)?;
            }
            ColumnData::Utf8(values) => {
                d.set_item("dtype", "string")?;
                d.set_item("values", values)?;
            }
        }
        out.append(d)?;
    }
    Ok(out.into())
}

fn profiles_to_py(py: Python<'_>, profiles: &[profile::ColumnProfile]) -> PyResult<PyObject> {
    let out = PyList::empty_bound(py);
    for p in profiles {
        let d = PyDict::new_bound(py);
        d.set_item("name", &p.name)?;
        d.set_item("null_count", p.null_count)?;
        d.set_item("unique_count", p.unique_count)?;
        d.set_item("min", p.min)?;
        d.set_item("max", p.max)?;
        d.set_item("mean", p.mean)?;
        out.append(d)?;
    }
    Ok(out.into())
}

fn get_bool(cfg: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    Ok(cfg.get_item(key)?.map_or(default, |v| v.extract().unwrap_or(default)))
}

fn get_float(cfg: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    Ok(cfg.get_item(key)?.map_or(default, |v| v.extract().unwrap_or(default)))
}

fn get_string(cfg: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    Ok(cfg
        .get_item(key)?
        .map_or_else(|| default.to_string(), |v| v.extract().unwrap_or_else(|_| default.to_string())))
}

fn get_optional_string(cfg: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match cfg.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
        _ => Ok(None),
    }
}

fn extract_strings(cfg: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    match cfg.get_item(key)? {
        Some(v) => v.extract(),
        None => Ok(Vec::new()),
    }
}

fn extract_pairs(cfg: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<(String, String)>> {
    match cfg.get_item(key)? {
        Some(v) => v.extract(),
        None => Ok(Vec::new()),
    }
}
