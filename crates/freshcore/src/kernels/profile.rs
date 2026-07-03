use std::collections::HashSet;

use crate::arrays::{CellKey, ColumnData, Frame};

#[derive(Clone, Debug, PartialEq)]
pub struct ColumnProfile {
    pub name: String,
    pub null_count: usize,
    pub unique_count: usize,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
}

pub fn profile(frame: &Frame) -> Vec<ColumnProfile> {
    frame
        .columns
        .iter()
        .map(|column| {
            let mut unique = HashSet::new();
            for row in 0..frame.nrows {
                unique.insert(column.data.cell_key(row));
            }
            let (min, max, mean) = numeric_stats(&column.data);
            ColumnProfile {
                name: column.name.clone(),
                null_count: column.data.null_count(),
                unique_count: unique
                    .into_iter()
                    .filter(|k| !matches!(k, CellKey::Null))
                    .count(),
                min,
                max,
                mean,
            }
        })
        .collect()
}

fn numeric_stats(data: &ColumnData) -> (Option<f64>, Option<f64>, Option<f64>) {
    let ColumnData::Float(values) = data else {
        return (None, None, None);
    };
    let vals: Vec<f64> = values.iter().flatten().copied().filter(|v| !v.is_nan()).collect();
    if vals.is_empty() {
        return (None, None, None);
    }
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    (Some(min), Some(max), Some(mean))
}
