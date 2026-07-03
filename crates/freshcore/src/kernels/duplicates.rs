use std::collections::HashSet;

use crate::arrays::{CellKey, Frame};

pub fn drop_duplicates(frame: &mut Frame, keep_policy: &str) -> usize {
    if frame.nrows == 0 {
        return 0;
    }
    let mut duplicated = vec![false; frame.nrows];
    if keep_policy == "last" {
        let mut seen: HashSet<Vec<CellKey>> = HashSet::new();
        for row in (0..frame.nrows).rev() {
            let key = row_key(frame, row);
            if !seen.insert(key) {
                duplicated[row] = true;
            }
        }
    } else {
        let mut seen: HashSet<Vec<CellKey>> = HashSet::new();
        for row in 0..frame.nrows {
            let key = row_key(frame, row);
            if !seen.insert(key) {
                duplicated[row] = true;
            }
        }
    }
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
}
