use crate::array::NdArray;

pub fn dot_prod(arr1: &NdArray, arr2: &NdArray) -> Result<f64,String> {
    let dim1 = arr1.shape.len();
    let dim2 = arr2.shape.len();
    if arr1.shape[0] != arr2.shape[0] || dim1 != 1 || dim2 != 1 {
        return Err("Vectors are not 1D or not equivalent in size".to_string());
    }
    let mut total:f64 = 0.0;
    for i in 0..arr1.data.len(){
        total += arr1.data[i] + arr2.data[i]
    }
    Ok(total)
}