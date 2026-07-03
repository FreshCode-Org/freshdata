use std::hash::{Hash, Hasher};

#[derive(Clone, Debug, PartialEq)]
pub enum ColumnData {
    Float(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
    Utf8(Vec<Option<String>>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    pub name: String,
    pub data: ColumnData,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub columns: Vec<Column>,
    pub nrows: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CellKey {
    Null,
    Float(u64),
    Bool(bool),
    Utf8(String),
}

impl Hash for CellKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            CellKey::Null => 0u8.hash(state),
            CellKey::Float(v) => {
                1u8.hash(state);
                v.hash(state);
            }
            CellKey::Bool(v) => {
                2u8.hash(state);
                v.hash(state);
            }
            CellKey::Utf8(v) => {
                3u8.hash(state);
                v.hash(state);
            }
        }
    }
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Float(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::Utf8(v) => v.len(),
        }
    }

    pub fn null_count(&self) -> usize {
        match self {
            ColumnData::Float(v) => v.iter().filter(|x| x.is_none()).count(),
            ColumnData::Bool(v) => v.iter().filter(|x| x.is_none()).count(),
            ColumnData::Utf8(v) => v.iter().filter(|x| x.is_none()).count(),
        }
    }

    pub fn is_all_null(&self) -> bool {
        self.null_count() == self.len()
    }

    pub fn cell_key(&self, row: usize) -> CellKey {
        match self {
            ColumnData::Float(v) => v[row]
                .map(|x| {
                    if x == 0.0 {
                        CellKey::Float(0.0f64.to_bits())
                    } else {
                        CellKey::Float(x.to_bits())
                    }
                })
                .unwrap_or(CellKey::Null),
            ColumnData::Bool(v) => v[row].map(CellKey::Bool).unwrap_or(CellKey::Null),
            ColumnData::Utf8(v) => v[row]
                .as_ref()
                .map(|x| CellKey::Utf8(x.clone()))
                .unwrap_or(CellKey::Null),
        }
    }

    pub fn take(&self, keep: &[bool]) -> ColumnData {
        match self {
            ColumnData::Float(v) => ColumnData::Float(
                v.iter()
                    .zip(keep)
                    .filter_map(|(value, keep)| keep.then_some(*value))
                    .collect(),
            ),
            ColumnData::Bool(v) => ColumnData::Bool(
                v.iter()
                    .zip(keep)
                    .filter_map(|(value, keep)| keep.then_some(*value))
                    .collect(),
            ),
            ColumnData::Utf8(v) => ColumnData::Utf8(
                v.iter()
                    .zip(keep)
                    .filter_map(|(value, keep)| keep.then_some(value.clone()))
                    .collect(),
            ),
        }
    }
}

impl Frame {
    pub fn new(columns: Vec<Column>) -> Result<Self, String> {
        let nrows = columns.first().map(|c| c.data.len()).unwrap_or(0);
        if columns.iter().any(|c| c.data.len() != nrows) {
            return Err("all columns must have the same length".to_string());
        }
        Ok(Self { columns, nrows })
    }

    pub fn null_cells(&self) -> usize {
        self.columns.iter().map(|c| c.data.null_count()).sum()
    }

    pub fn take_rows(&mut self, keep: &[bool]) {
        for column in &mut self.columns {
            column.data = column.data.take(keep);
        }
        self.nrows = keep.iter().filter(|x| **x).count();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_counts_are_separate_from_values() {
        let data = ColumnData::Float(vec![Some(1.0), None, Some(2.0)]);
        assert_eq!(data.len(), 3);
        assert_eq!(data.null_count(), 1);
        assert!(!data.is_all_null());
    }

    #[test]
    fn row_take_preserves_column_lengths() {
        let mut frame = Frame::new(vec![Column {
            name: "a".into(),
            data: ColumnData::Utf8(vec![Some("x".into()), None, Some("y".into())]),
        }])
        .unwrap();
        frame.take_rows(&[true, false, true]);
        assert_eq!(frame.nrows, 2);
        assert_eq!(frame.columns[0].data.len(), 2);
    }
}
