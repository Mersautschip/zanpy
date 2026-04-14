use pyo3::prelude::*;
use zanpy_core::array::NdArray;

#[pyclass]
struct PyNdArray {
    inner: NdArray
}

#[pymethods]
impl PyNdArray {
    #[new]
    fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        // 1. PyO3 creates 'shape' as a Vec<usize> from the Python list.
        // 2. We borrow it as a slice '&shape' to satisfy zanpy-core.
        PyNdArray { 
            inner: NdArray::new(data, &shape) 
        }
    }

    fn get(&self, indices: Vec<usize>) -> PyResult<f64> {
        // 1. PyO3 creates 'indices' as a Vec<usize>.
        // 2. We borrow it as '&indices' for the .get() method.
        self.inner.get(&indices).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(e)
        })
    }
}


#[pymodule]
fn zanpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNdArray>()?;
    Ok(())
}