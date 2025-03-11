use pyo3::prelude::*;
use numpy::{PyArray, IntoPyArray};
use ndarray::{ArrayD, ArrayViewD};
use std::collections::HashMap;

/// Main function for tensor optimization
#[pyfunction]
fn optimize_tensors(
    py: Python,
    tensors: HashMap<String, PyObject>,
) -> PyResult<HashMap<String, PyObject>> {
    let mut optimized: HashMap<String, PyObject> = HashMap::new();
    
    for (name, tensor) in tensors {
        // Convert tensor to multi-dimensional array
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

/// Optimize multi-dimensional array
fn optimize_array(array: &mut ArrayD<f32>) {
    array.mapv_inplace(|x| {
        if x.abs() < 1e-6 {
            0.0
        } else {
            x
        }
    });
}

/// Initialize Python module
#[pymodule]
fn evaopt_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_tensors, m)?)?;
    Ok(())
} 