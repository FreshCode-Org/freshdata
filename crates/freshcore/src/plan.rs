use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StringCase {
    Preserve,
    Lower,
    Upper,
}

#[derive(Clone, Debug)]
pub struct CleanPlan {
    pub rename_map: Vec<(String, String)>,
    pub strip_whitespace: bool,
    pub normalize_sentinels: bool,
    pub sentinels: HashSet<String>,
    pub string_case: StringCase,
    pub drop_empty_columns: bool,
    pub drop_empty_rows: bool,
    pub drop_duplicates: bool,
    pub duplicate_keep: String,
    pub fix_dtypes: bool,
    pub numeric_threshold: f64,
    pub preserve_leading_zeros: bool,
    pub impute: Option<String>,
    pub outliers: Option<String>,
    pub outlier_method: String,
    pub outlier_factor: f64,
}

impl Default for CleanPlan {
    fn default() -> Self {
        Self {
            rename_map: Vec::new(),
            strip_whitespace: true,
            normalize_sentinels: true,
            sentinels: HashSet::new(),
            string_case: StringCase::Preserve,
            drop_empty_columns: true,
            drop_empty_rows: true,
            drop_duplicates: true,
            duplicate_keep: "first".to_string(),
            fix_dtypes: true,
            numeric_threshold: 0.95,
            preserve_leading_zeros: true,
            impute: None,
            outliers: None,
            outlier_method: "iqr".to_string(),
            outlier_factor: 1.5,
        }
    }
}
