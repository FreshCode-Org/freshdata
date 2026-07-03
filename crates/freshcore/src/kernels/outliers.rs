use crate::arrays::{Column, ColumnData, Frame};

#[derive(Clone, Debug, PartialEq)]
pub struct OutlierAction {
    pub column: String,
    pub action: String,
    pub count: usize,
    pub lower: f64,
    pub upper: f64,
    pub flag_column: Option<String>,
}

pub fn handle_outliers(
    frame: &mut Frame,
    action: Option<&str>,
    method: &str,
    factor: f64,
) -> Vec<OutlierAction> {
    let Some(action) = action else {
        return Vec::new();
    };
    let mut actions = Vec::new();
    let existing: Vec<String> = frame.columns.iter().map(|c| c.name.clone()).collect();
    let mut flags = Vec::new();

    for column in &mut frame.columns {
        let ColumnData::Float(values) = &mut column.data else {
            continue;
        };
        let Some((lo, hi)) = bounds(values, method, factor) else {
            continue;
        };
        let mask: Vec<bool> = values
            .iter()
            .map(|v| v.is_some_and(|x| x < lo || x > hi))
            .collect();
        let count = mask.iter().filter(|v| **v).count();
        if count == 0 {
            continue;
        }
        if action == "clip" {
            for value in values.iter_mut().flatten() {
                if *value < lo {
                    *value = lo;
                } else if *value > hi {
                    *value = hi;
                }
            }
            actions.push(OutlierAction {
                column: column.name.clone(),
                action: "clip".into(),
                count,
                lower: lo,
                upper: hi,
                flag_column: None,
            });
        } else {
            let flag = unique_flag(&existing, &format!("{}_outlier", column.name));
            flags.push(Column {
                name: flag.clone(),
                data: ColumnData::Bool(mask.into_iter().map(Some).collect()),
            });
            actions.push(OutlierAction {
                column: column.name.clone(),
                action: "flag".into(),
                count,
                lower: lo,
                upper: hi,
                flag_column: Some(flag),
            });
        }
    }
    if !flags.is_empty() {
        frame.columns.extend(flags);
    }
    actions
}

pub fn bounds(values: &[Option<f64>], method: &str, factor: f64) -> Option<(f64, f64)> {
    let vals: Vec<f64> = values.iter().flatten().copied().filter(|v| !v.is_nan()).collect();
    if vals.len() < 2 {
        return None;
    }
    if method == "zscore" {
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
        let std = var.sqrt();
        if std == 0.0 || std.is_nan() {
            return None;
        }
        return Some((mean - factor * std, mean + factor * std));
    }
    let q1 = quantile(vals.clone(), 0.25)?;
    let q3 = quantile(vals, 0.75)?;
    let spread = q3 - q1;
    if spread == 0.0 || spread.is_nan() {
        return None;
    }
    Some((q1 - factor * spread, q3 + factor * spread))
}

fn quantile(mut vals: Vec<f64>, q: f64) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let pos = q * (vals.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        Some(vals[lo])
    } else {
        let weight = pos - lo as f64;
        Some(vals[lo] * (1.0 - weight) + vals[hi] * weight)
    }
}

fn unique_flag(existing: &[String], base: &str) -> String {
    let mut name = base.to_string();
    let mut i = 1usize;
    while existing.contains(&name) {
        i += 1;
        name = format!("{base}_{i}");
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iqr_bounds_detect_extreme_value() {
        let values = vec![Some(1.0), Some(2.0), Some(3.0), Some(100.0)];
        let (lo, hi) = bounds(&values, "iqr", 1.5).unwrap();
        assert!(lo < 1.0);
        assert!(hi < 100.0);
    }
}
