use rayon::prelude::*;

use crate::arrays::{ColumnData, Frame};
use crate::plan::{CleanPlan, StringCase};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StringAction {
    pub column: String,
    pub trimmed: usize,
    pub sentinels: usize,
    pub case_normalized: usize,
}

pub fn clean_strings(frame: &mut Frame, plan: &CleanPlan) -> Vec<StringAction> {
    frame
        .columns
        .par_iter_mut()
        .filter_map(|column| match &mut column.data {
            ColumnData::Utf8(values) => {
                let mut action = StringAction {
                    column: column.name.clone(),
                    ..StringAction::default()
                };
                for slot in values {
                    let Some(value) = slot.as_mut() else {
                        continue;
                    };
                    if plan.strip_whitespace {
                        let stripped = value.trim();
                        if stripped.len() != value.len() {
                            *value = stripped.to_string();
                            action.trimmed += 1;
                        }
                    }
                    if plan.normalize_sentinels
                        && plan.sentinels.contains(&value.trim().to_lowercase())
                    {
                        *slot = None;
                        action.sentinels += 1;
                        continue;
                    }
                    match plan.string_case {
                        StringCase::Preserve => {}
                        StringCase::Lower => {
                            let lowered = value.to_lowercase();
                            if lowered != *value {
                                *value = lowered;
                                action.case_normalized += 1;
                            }
                        }
                        StringCase::Upper => {
                            let uppered = value.to_uppercase();
                            if uppered != *value {
                                *value = uppered;
                                action.case_normalized += 1;
                            }
                        }
                    }
                }
                (action.trimmed > 0 || action.sentinels > 0 || action.case_normalized > 0)
                    .then_some(action)
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::arrays::{Column, ColumnData, Frame};

    #[test]
    fn trims_and_nulls_sentinels() {
        let mut frame = Frame::new(vec![Column {
            name: "name".into(),
            data: ColumnData::Utf8(vec![Some(" x ".into()), Some("N/A".into())]),
        }])
        .unwrap();
        let mut plan = CleanPlan::default();
        plan.sentinels = HashSet::from(["n/a".to_string()]);
        let actions = clean_strings(&mut frame, &plan);
        assert_eq!(actions[0].trimmed, 1);
        assert_eq!(actions[0].sentinels, 1);
    }
}
