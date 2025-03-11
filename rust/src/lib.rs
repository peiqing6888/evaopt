use pyo3::prelude::*;
use numpy::{PyArray, IntoPyArray};
use ndarray::{ArrayD, ArrayViewD};
use std::collections::HashMap;

/// 優化張量的主要函數
#[pyfunction]
fn optimize_tensors(
    py: Python,
    tensors: HashMap<String, PyObject>,
) -> PyResult<HashMap<String, PyObject>> {
    let mut optimized: HashMap<String, PyObject> = HashMap::new();
    
    for (name, tensor) in tensors {
        // 將張量轉換為多維數組
        if let Ok(array) = tensor.extract::<&PyArray<f32, _>>(py) {
            let readonly = array.readonly();
            let view = readonly.as_array();
            let mut optimized_array: ArrayD<f32> = view.to_owned();
            optimize_array(&mut optimized_array);
            let py_array = optimized_array.into_pyarray(py);
            optimized.insert(name, py_array.to_object(py));
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported tensor type"
            ));
        }
    }
    
    Ok(optimized)
}

/// 優化多維數組
fn optimize_array(array: &mut ArrayD<f32>) {
    array.mapv_inplace(|x| {
        if x.abs() < 1e-6 {
            0.0
        } else {
            x
        }
    });
}

/// 初始化 Python 模塊
#[pymodule]
fn evaopt_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_tensors, m)?)?;
    Ok(())
} 