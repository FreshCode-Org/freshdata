use std::collections::HashMap;

use crate::arrays::{ColumnData, Frame};

#[derive(Clone, Debug, PartialEq)]
pub struct ImputeAction {
    pub column: String,
    pub strategy: String,
    pub count: usize,
    pub value: String,
}

pub fn drop_empty_columns(frame: &mut Frame) -> Vec<String> {
    let mut dropped = Vec::new();
    frame.columns.retain(|column| {
        let keep = !column.data.is_all_null();
        if !keep {
            dropped.push(column.name.clone());
        }
        keep
    });
    dropped
}

pub fn drop_empty_rows(frame: &mut Frame) -> usize {
    if frame.nrows == 0 {
        return 0;
    }
    let mut keep = vec![false; frame.nrows];
    for row in 0..frame.nrows {
        keep[row] = frame
            .columns
            .iter()
            .any(|column| !matches!(column.data.cell_key(row), crate::arrays::CellKey::Null));
    }
    let dropped = keep.iter().filter(|v| !**v).count();
    if dropped > 0 {
        frame.take_rows(&keep);
    }
    dropped
}

pub fn impute(frame: &mut Frame, strategy: Option<&str>) -> Vec<ImputeAction> {
    let Some(strategy) = strategy else {
        return Vec::new();
    };
    let mut actions = Vec::new();
    for column in &mut frame.columns {
        match &mut column.data {
            ColumnData::Float(values) => {
                let missing = values.iter().filter(|v| v.is_none()).count();
                if missing == 0 || missing == values.len() {
                    continue;
                }
                let resolved = if strategy == "auto" { "median" } else { strategy };
                let fill = match resolved {
                    "mean" => mean(values),
                    "median" => median(values),
                    "mode" => mode_float(values),
                    _ => None,
                };
                if let Some(fill) = fill {
                    for value in values.iter_mut().filter(|v| v.is_none()) {
                        *value = Some(fill);
                    }
                    actions.push(ImputeAction {
                        column: column.name.clone(),
                        strategy: resolved.to_string(),
                        count: missing,
                        value: format!("{fill:.6}"),
                    });
                }
            }
            ColumnData::Utf8(values) => {
                let missing = values.iter().filter(|v| v.is_none()).count();
                if missing == 0 || missing == values.len() {
                    continue;
                }
                let resolved = if strategy == "auto" { "mode" } else { strategy };
                if resolved != "mode" {
                    continue;
                }
                if let Some(fill) = mode_string(values) {
                    for value in values.iter_mut().filter(|v| v.is_none()) {
                        *value = Some(fill.clone());
                    }
                    actions.push(ImputeAction {
                        column: column.name.clone(),
                        strategy: "mode".to_string(),
                        count: missing,
                        value: format!("{fill:?}"),
                    });
                }
            }
            ColumnData::Bool(_) => {}
        }
    }
    actions
}

fn mean(values: &[Option<f64>]) -> Option<f64> {
    let mut sum = 0.0;
    let mut n = 0usize;
    for v in values.iter().flatten() {
        sum += *v;
        n += 1;
    }
    (n > 0).then_some(sum / n as f64)
}

fn median(values: &[Option<f64>]) -> Option<f64> {
    let mut vals: Vec<f64> = values.iter().flatten().copied().filter(|v| !v.is_nan()).collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 0 {
        Some((vals[mid - 1] + vals[mid]) / 2.0)
    } else {
        Some(vals[mid])
    }
}

fn mode_float(values: &[Option<f64>]) -> Option<f64> {
    let mut counts: HashMap<u64, (f64, usize)> = HashMap::new();
    for v in values.iter().flatten().filter(|v| !v.is_nan()) {
        counts
            .entry(v.to_bits())
            .and_modify(|(_, n)| *n += 1)
            .or_insert((*v, 1));
    }
    counts
        .values()
        .max_by(|(av, an), (bv, bn)| an.cmp(bn).then_with(|| bv.total_cmp(av)))
        .map(|(v, _)| *v)
}

fn mode_string(values: &[Option<String>]) -> Option<String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for value in values.iter().flatten() {
        *counts.entry(value.as_str()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by(|(av, an), (bv, bn)| an.cmp(bn).then_with(|| bv.cmp(av)))
        .map(|(v, _)| v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::{Column, ColumnData, Frame};

    #[test]
    fn median_impute_fills_missing_numeric() {
        let mut frame = Frame::new(vec![Column {
            name: "x".into(),
            data: ColumnData::Float(vec![Some(1.0), None, Some(3.0)]),
        }])
        .unwrap();
        let actions = impute(&mut frame, Some("median"));
        assert_eq!(actions[0].count, 1);
        assert_eq!(
            frame.columns[0].data,
            ColumnData::Float(vec![Some(1.0), Some(2.0), Some(3.0)])
        );
    }
}
