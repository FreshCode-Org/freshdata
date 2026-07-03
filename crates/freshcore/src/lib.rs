mod arrays;
mod kernels;
mod plan;
mod python;

pub use arrays::{Column, ColumnData, Frame};
pub use plan::{CleanPlan, StringCase};

use pyo3::prelude::*;

#[pymodule]
fn freshdata_freshcore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
