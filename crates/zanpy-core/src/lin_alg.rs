use crate::array::NdArray;

pub fn mat_mul(arr1: &NdArray, arr2: &NdArray) -> Result<NdArray,String> {
    // We're bounding this to 2D
    // This is the naive approach, add tiling later

    let rows_a = arr1.shape[0];
    let cols_a = arr1.shape[1];
    let rows_b = arr2.shape[0];
    let cols_b = arr2.shape[1];

    if cols_a != rows_b {
        return Err("Error: Matrix 1 Column Does Not Match Matrix 2 Rows".to_string())
    }

    if arr1.shape.len() != 2 || arr2.shape.len() != 2{
        return Err("Error: Only 2D Matrices Are Allowed".to_string())
    }

    let mut result_data = vec![0.0; rows_a * cols_b];

    

    let s1_row = arr1.stride[0];
    let s1_col = arr1.stride[1];
    let s2_row = arr2.stride[0];
    let s2_col = arr2.stride[1];

    for i in 0..rows_a {
        for k in 0..cols_a {
            // Using the stride of arr1 to find the element
            let val_a = arr1.data[i * s1_row + k * s1_col];

            for j in 0..cols_b {
                // We calculate the flat index for the result and arr2
                // Result is always a fresh, contiguous Vec (Standard Row-Major)
                // let res_idx = i * cols_b + j; Removed for speed 
                // let arr2_idx = k * s2_row + j * s2_col; Same

                // arr2 might have a custom stride (e.g. if it was transposed)
                
                result_data[i * cols_b + j] += val_a * arr2.data[k * s2_row + j * s2_col];
            }
        }
    }

    Ok(NdArray::new(result_data, vec![rows_a,cols_b]))

}

