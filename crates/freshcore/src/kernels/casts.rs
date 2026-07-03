use rayon::prelude::*;

use crate::arrays::{Column, ColumnData, Frame};
use crate::plan::CleanPlan;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CastAction {
    pub column: String,
    pub target: String,
    pub count: usize,
    pub coerced: usize,
}

const TRUE_WORDS: &[&str] = &["true", "t", "yes", "y"];
const FALSE_WORDS: &[&str] = &["false", "f", "no", "n"];

pub fn infer_and_cast(frame: &mut Frame, plan: &CleanPlan) -> Vec<CastAction> {
    if !plan.fix_dtypes {
        return Vec::new();
    }
    frame
        .columns
        .par_iter_mut()
        .filter_map(|column| cast_column(column, plan))
        .collect()
}

fn cast_column(column: &mut Column, plan: &CleanPlan) -> Option<CastAction> {
    let ColumnData::Utf8(values) = &column.data else {
        return None;
    };
    let non_null: Vec<&str> = values.iter().filter_map(|v| v.as_deref()).collect();
    if non_null.is_empty() {
        return None;
    }
    if let Some(bools) = try_bool(values) {
        let count = bools.iter().filter(|v| v.is_some()).count();
        column.data = ColumnData::Bool(bools);
        return Some(CastAction {
            column: column.name.clone(),
            target: "boolean".into(),
            count,
            coerced: 0,
        });
    }
    if plan.preserve_leading_zeros && non_null.iter().any(|v| has_leading_zero(v)) {
        return None;
    }
    let (parsed, parsed_count, coerced) = parse_numeric(values);
    if (parsed_count as f64) / (non_null.len() as f64) >= plan.numeric_threshold {
        column.data = ColumnData::Float(parsed);
        return Some(CastAction {
            column: column.name.clone(),
            target: "float64".into(),
            count: parsed_count + coerced,
            coerced,
        });
    }
    None
}

fn try_bool(values: &[Option<String>]) -> Option<Vec<Option<bool>>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let Some(raw) = value else {
            out.push(None);
            continue;
        };
        let token = raw.trim().to_lowercase();
        if TRUE_WORDS.contains(&token.as_str()) {
            out.push(Some(true));
        } else if FALSE_WORDS.contains(&token.as_str()) {
            out.push(Some(false));
        } else {
            return None;
        }
    }
    Some(out)
}

fn parse_numeric(values: &[Option<String>]) -> (Vec<Option<f64>>, usize, usize) {
    let mut out = Vec::with_capacity(values.len());
    let mut parsed = 0usize;
    let mut coerced = 0usize;
    for value in values {
        let Some(raw) = value else {
            out.push(None);
            continue;
        };
        let cleaned = raw
            .trim()
            .trim_matches('$')
            .trim_matches('€')
            .trim_matches('£')
            .trim_matches('₹')
            .replace(',', "");
        match cleaned.parse::<f64>() {
            Ok(v) => {
                out.push(Some(v));
                parsed += 1;
            }
            Err(_) => {
                out.push(None);
                coerced += 1;
            }
        }
    }
    (out, parsed, coerced)
}

fn has_leading_zero(value: &str) -> bool {
    let trimmed = value.trim_start_matches(['+', '-']).trim();
    trimmed.len() > 1
        && trimmed.starts_with('0')
        && trimmed.chars().nth(1).is_some_and(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::{Column, ColumnData, Frame};

    #[test]
    fn numeric_cast_preserves_zero_padded_ids() {
        let mut frame = Frame::new(vec![Column {
            name: "zip".into(),
            data: ColumnData::Utf8(vec![Some("02115".into()), Some("10001".into())]),
        }])
        .unwrap();
        assert!(infer_and_cast(&mut frame, &CleanPlan::default()).is_empty());
    }

    #[test]
    fn bool_cast_uses_small_vocabulary() {
        let mut frame = Frame::new(vec![Column {
            name: "flag".into(),
            data: ColumnData::Utf8(vec![Some("yes".into()), Some("no".into())]),
        }])
        .unwrap();
        let actions = infer_and_cast(&mut frame, &CleanPlan::default());
        assert_eq!(actions[0].target, "boolean");
    }
}
