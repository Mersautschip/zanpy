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

    if arr1.rank != 2 || arr2.rank != 2{
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

                // Function does not currently support transposed matrices
                
                result_data[i * cols_b + j] += val_a * arr2.data[k * s2_row + j * s2_col];
            }
        }
    }

    // This might be a time-loss:
    let mut input_shape = [1usize;8];
    input_shape[0] = rows_a;
    input_shape[1] = cols_b;

    Ok(NdArray::new(result_data, &input_shape))

}

fn offset(stride: &[usize;8], indices: [usize;2]) -> usize {
    //Initializing the offset
    let mut offset: usize = 0;

    //Basically applying the concept explained in new
    for i in 0..stride.len(){
        offset = offset + (stride[i]*indices[i])
    }
    offset
}
// Probably should flag this for inline
fn get_2d(data: &Vec<f64>, stride: &[usize;8], indices: [usize;2]) -> f64 {
    data[offset(stride, indices)]
}

pub fn inverse(arr: &NdArray) -> Result<NdArray, String> {
    // This will use LU decomposition
    let stride = &arr.stride;
    let n = arr.shape[0];
    if arr.shape.len() != 2 || n != arr.shape[1]{
        return Err("Matrix must be a square & two dimensional".to_string());
    }

    let mut a = arr.data.clone();
    let mut inv_data = NdArray::identity(n);
    
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&i, &j|{
                get_2d(&a, stride, [i,col])
                    .abs()
                    .partial_cmp(
                        &get_2d(&a, stride, [j,col])
                            .abs()
                    )
                    .unwrap()
            })
            .unwrap();
        if get_2d(&a, stride, [pivot_row,col]).abs() < 1e-12{
            return Err("Matrix is singular and cannot be inverted!".to_string());
        }

        if pivot_row != col {
            for k in 0..n {
                a.swap(
                    offset(stride,[col,k]),
                    offset(stride,[pivot_row,k]) 
                );
                inv_data.data.swap(
                    col * n + k,
                    pivot_row * n + k
                );
            }
        }

        let pivot_val = a[offset(stride,[col, col])];
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[offset(stride,[row, col])] / pivot_val;
            for k in 0..n {
                let a_val = a[offset(stride,[col, k])];
                a[offset(stride,[row, k])] -= factor * a_val;
                let inv_val = inv_data.get(&[col, k,0,0,0,0,0,0])?;
                inv_data.data[row*n+k] -= factor * inv_val;
            }
        }

        for k in 0..n {
            a[offset(stride,[col,k])] /= pivot_val;
            inv_data.data[col*n+k] /= pivot_val;
        }
    }
    Ok(inv_data)
}