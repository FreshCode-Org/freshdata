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
    let q3 = quantile(vals.clone(), 0.75)?;
    let spread = q3 - q1;
    if spread.is_nan() {
        return None;
    }
    if spread == 0.0 {
        return zero_iqr_bounds(&vals, q1, factor);
    }
    Some((q1 - factor * spread, q3 + factor * spread))
}

/// IQR-equivalent spread per unit of mean absolute deviation for normal data:
/// sqrt(pi/2) (MeanAD -> sigma) x 2*Phi^-1(0.75) (sigma -> IQR). Mirrors
/// `freshdata.steps.outliers.MEANAD_TO_IQR`.
const MEANAD_TO_IQR: f64 = 1.6906950787902986;
/// Largest share of values the zero-IQR fallback may flag; mirrors
/// `freshdata.steps.outliers.ZERO_IQR_MAX_SHARE`.
const ZERO_IQR_MAX_SHARE: f64 = 0.05;

/// Fences when both quartiles equal `center` (at least half the column sits on
/// one value). A zero IQR must not silently disable detection. Instead, the IQR
/// is replaced by the normal-consistent spread from the mean absolute deviation
/// around the median (the median equals the quartiles here). The result is
/// `None` for a constant column, or when the fences would flag more than
/// `ZERO_IQR_MAX_SHARE` of the values: that many is a second mode, not rare
/// outliers.
fn zero_iqr_bounds(vals: &[f64], center: f64, factor: f64) -> Option<(f64, f64)> {
    let n = vals.len() as f64;
    let mean_ad = vals.iter().map(|v| (v - center).abs()).sum::<f64>() / n;
    if mean_ad == 0.0 || mean_ad.is_nan() {
        return None;
    }
    let spread = MEANAD_TO_IQR * mean_ad;
    let (lo, hi) = (center - factor * spread, center + factor * spread);
    let flagged = vals.iter().filter(|v| **v < lo || **v > hi).count() as f64;
    if flagged > ZERO_IQR_MAX_SHARE * n {
        return None;
    }
    Some((lo, hi))
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

    fn zeros_with(extra: &[f64], zeros: usize) -> Vec<Option<f64>> {
        std::iter::repeat(0.0)
            .take(zeros)
            .chain(extra.iter().copied())
            .map(Some)
            .collect()
    }

    #[test]
    fn zero_iqr_falls_back_to_mean_absolute_deviation() {
        let values = zeros_with(&[1000.0, 5000.0, 2.0, 3.0, -800.0], 95);
        let (lo, hi) = bounds(&values, "iqr", 1.5).unwrap();
        // MeanAD = 6805 / 100; fences = 0 +/- 1.5 x 1.6907 x MeanAD.
        let expected = 1.5 * (MEANAD_TO_IQR * (6805.0 / 100.0));
        assert_eq!((lo, hi), (-expected, expected));
        let mut frame = Frame {
            nrows: values.len(),
            columns: vec![Column {
                name: "x".into(),
                data: ColumnData::Float(values),
            }],
        };
        let actions = handle_outliers(&mut frame, Some("clip"), "iqr", 1.5);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].count, 3); // the spikes, not 2.0 / 3.0
    }

    #[test]
    fn zero_iqr_fallback_does_not_flag_a_second_mode() {
        // 30% non-zero, split around zero: the IQR is still zero.
        let extra: Vec<f64> = (1..=15).flat_map(|i| [-(i as f64), i as f64]).collect();
        assert_eq!(bounds(&zeros_with(&extra, 70), "iqr", 1.5), None);
    }

    #[test]
    fn constant_column_has_no_bounds() {
        let values = vec![Some(5.0); 20];
        assert_eq!(bounds(&values, "iqr", 1.5), None);
        assert_eq!(bounds(&values, "zscore", 3.0), None);
    }
}
