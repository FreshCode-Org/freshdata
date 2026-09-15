use std::collections::HashSet;

use crate::arrays::{CellKey, Frame};

/// Marks full-row duplicates under `keep_policy` (`"last"` keeps the final
/// occurrence, anything else the first).
///
/// Like pandas' `drop_duplicate_rows`, a frame with no rows or no columns has
/// no duplicates.
pub fn duplicated_mask(frame: &Frame, keep_policy: &str) -> Vec<bool> {
    let mut duplicated = vec![false; frame.nrows];
    if frame.nrows == 0 || frame.columns.is_empty() {
        return duplicated;
    }
    let mut seen: HashSet<Vec<CellKey>> = HashSet::new();
    let mut mark = |row: usize| {
        if !seen.insert(row_key(frame, row)) {
            duplicated[row] = true;
        }
    };
    if keep_policy == "last" {
        (0..frame.nrows).rev().for_each(&mut mark);
    } else {
        (0..frame.nrows).for_each(&mut mark);
    }
    duplicated
}

/// Number of full-row duplicates, without removing any rows.
///
/// The count does not depend on the keep policy: every row beyond the first
/// occurrence of its key is a duplicate.
pub fn count_duplicates(frame: &Frame) -> usize {
    duplicated_mask(frame, "first")
        .iter()
        .filter(|v| **v)
        .count()
}

pub fn drop_duplicates(frame: &mut Frame, keep_policy: &str) -> usize {
    let duplicated = duplicated_mask(frame, keep_policy);
    let dropped = duplicated.iter().filter(|v| **v).count();
    if dropped > 0 {
        let keep: Vec<bool> = duplicated.iter().map(|v| !*v).collect();
        frame.take_rows(&keep);
    }
    dropped
}

fn row_key(frame: &Frame, row: usize) -> Vec<CellKey> {
    frame
        .columns
        .iter()
        .map(|column| column.data.cell_key(row))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::{Column, ColumnData, Frame};

    fn mixed_frame() -> Frame {
        Frame::new(vec![
            Column {
                name: "a".into(),
                data: ColumnData::Float(vec![
                    Some(1.0),
                    Some(1.0),
                    Some(0.0),
                    Some(-0.0),
                    None,
                    None,
                    Some(2.0),
                ]),
            },
            Column {
                name: "b".into(),
                data: ColumnData::Utf8(vec![
                    Some("x".into()),
                    Some("x".into()),
                    Some("y".into()),
                    Some("y".into()),
                    None,
                    None,
                    Some("x".into()),
                ]),
            },
        ])
        .unwrap()
    }

    #[test]
    fn removes_duplicate_rows() {
        let mut frame = Frame::new(vec![Column {
            name: "x".into(),
            data: ColumnData::Utf8(vec![Some("a".into()), Some("a".into()), Some("b".into())]),
        }])
        .unwrap();
        assert_eq!(drop_duplicates(&mut frame, "first"), 1);
        assert_eq!(frame.nrows, 2);
    }

    #[test]
    fn counts_duplicates_without_removing_rows() {
        let frame = mixed_frame();
        // (1.0, x) twice, (0.0, y) == (-0.0, y), and (null, null) twice.
        assert_eq!(count_duplicates(&frame), 3);
        assert_eq!(frame.nrows, 7);
        assert_eq!(frame, mixed_frame());
    }

    #[test]
    fn count_matches_rows_dropped_for_every_keep_policy() {
        for keep in ["first", "last"] {
            let mut frame = mixed_frame();
            let counted = count_duplicates(&frame);
            assert_eq!(drop_duplicates(&mut frame, keep), counted);
            assert_eq!(count_duplicates(&frame), 0);
        }
    }

    #[test]
    fn mask_marks_the_occurrences_the_keep_policy_drops() {
        let frame = mixed_frame();
        assert_eq!(
            duplicated_mask(&frame, "first"),
            vec![false, true, false, true, false, true, false]
        );
        assert_eq!(
            duplicated_mask(&frame, "last"),
            vec![true, false, true, false, true, false, false]
        );
    }

    #[test]
    fn frames_without_rows_or_columns_have_no_duplicates() {
        let no_rows = Frame::new(vec![Column {
            name: "x".into(),
            data: ColumnData::Float(Vec::new()),
        }])
        .unwrap();
        assert_eq!(count_duplicates(&no_rows), 0);

        let mut no_columns = Frame {
            columns: Vec::new(),
            nrows: 3,
        };
        assert_eq!(count_duplicates(&no_columns), 0);
        assert_eq!(drop_duplicates(&mut no_columns, "first"), 0);
        assert_eq!(no_columns.nrows, 3);
    }
}
