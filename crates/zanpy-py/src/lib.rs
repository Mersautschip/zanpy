use pyo3::prelude::*;
use zanpy_core::array::NdArray;
use zanpy_core::ops;
use zanpy_core::lin_alg;

#[pyclass]
struct PyNdArray {
    inner: NdArray
}

#[pymethods]
impl PyNdArray {

    #[new]
    fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        PyNdArray {
            inner: NdArray::new(data, &shape),
        }
    }

    #[staticmethod]
    fn ones(shape: Vec<usize>) -> Self {
        PyNdArray {
            inner: NdArray::ones(&shape),
        }
    }

    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        PyNdArray {
            inner: NdArray::zeros(&shape),
        }
    }

    #[staticmethod]
    fn arange(start: f64, end: f64, step: f64) -> Self {
        PyNdArray {
            inner: NdArray::arange(start, end, step),
        }
    }

    #[staticmethod]
    fn identity(x: usize) -> Self {
        PyNdArray {
            inner: NdArray::identity(x),
        }
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        // Return only the active dimensions based on rank
        self.inner.shape[..self.inner.rank].to_vec()
    }

    fn get(&self, indices: Vec<usize>) -> PyResult<f64> {
        // 1. PyO3 creates 'indices' as a Vec<usize>.
        // 2. We borrow it as '&indices' for the .get() method.
        self.inner.get(&indices).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(e)
        })
    }
    
    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Self> {
        let new_inner = NdArray::reshape(&self.inner, &new_shape)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyNdArray { inner: new_inner })
    }

    fn transpose(&mut self) -> PyResult<()> {
        self.inner.transpose()
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn permute(&self, axes: Vec<usize>) -> PyResult<Self> {
        let new_inner = self.inner.permute(&axes)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyNdArray { inner: new_inner })
    }

    fn __add__(&self, other: &PyNdArray) -> PyResult<Self> {
        let result = ops::add(&self.inner, &other.inner)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyNdArray { inner: result })
    }

    fn __sub__(&self, other: &PyNdArray) -> PyResult<Self> {
        let result = ops::subtract(&self.inner, &other.inner)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyNdArray { inner: result })
    }

    fn __mul__(&self, other: &PyNdArray) -> PyResult<Self> {
        let result = ops::multiply(&self.inner, &other.inner)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyNdArray { inner: result })
    }

    fn __truediv__(&self, other: &PyNdArray) -> PyResult<Self> {
        let result = ops::divide(&self.inner, &other.inner)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyNdArray { inner: result })
    }

    // --- LINEAR ALGEBRA & REDUCTIONS ---

    fn dot(&self, other: &PyNdArray) -> PyResult<f64> {
        ops::dot_prod(&self.inner, &other.inner)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn sum(&self) -> f64 {
        ops::sum(&self.inner)
    }

    fn mean(&self) -> f64 {
        ops::mean(&self.inner)
    }

    fn max(&self) -> f64 {
        ops::max(&self.inner)
    }

    fn min(&self) -> f64 {
        ops::min(&self.inner)
    }

    fn __matmul__(&self, arr2: &PyNdArray) -> PyResult<Self> {
        let result = lin_alg::mat_mul(&self.inner, &arr2.inner)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyNdArray { inner: result })
    }

    /// Matrix Inverse
    fn inv(&self) -> PyResult<Self> {
        // We use self.inner because &self is now in the function signature
        let result = lin_alg::inverse(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(PyNdArray { inner: result })
    }
}


#[pymodule]
fn zanpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNdArray>()?;
    Ok(())
}