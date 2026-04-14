use crate::array::NdArray;
use wide::f64x4;
use std::mem::MaybeUninit;

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

    let mut result_data: Vec<MaybeUninit<f64>> = Vec::with_capacity(rows_a * cols_b);

    unsafe {
        std::ptr::write_bytes(result_data.as_mut_ptr(), 0, rows_a * cols_b);
    }

    let s1_row = arr1.stride[0];
    let s1_col = arr1.stride[1];
    let s2_row = arr2.stride[0];
    let s2_col = arr2.stride[1];

    let a_ptr = arr1.data.as_ptr();
    let b_ptr = arr2.data.as_ptr();
    let r_ptr = result_data.as_mut_ptr() as *mut f64;

    for i in 0..rows_a {
        for k in 0..cols_a {
            // Using the stride of arr1 to find the element0
            let val_a = unsafe {*a_ptr.add(i * s1_row + k * s1_col)};
            let a_vec = f64x2::splat(val_a);

            let b_row_p = unsafe {b_ptr.add(k * s2_row)};
            let r_row_p = unsafe{r_ptr.add(i * cols_b)};

            let mut j = 0;
            while j + 2 <= cols_b {
                    unsafe {
                    let b_vec = f64x2::from([ *b_row_ptr.add(j), *b_row_ptr.add(j + 1) ]);
                    let r_vec = f64x2::from([ *r_row_ptr.add(j), *r_row_ptr.add(j + 1) ]);
                    let result = r_vec + a_vec * b_vec;
                    let out: [f64; 2] = result.into();
                    *r_row_ptr.add(j) = out[0];
                    *r_row_ptr.add(j + 1) = out[1];
                }
                j += 2;
            }
            
            
            if j < cols_b {
                unsafe {
                    *r_row_ptr.add(j) += val_a * *b_row_ptr.add(j);
                }
            }
        }
    }

    let result_data = unsafe{
        std::mem::transmute::<Vec<MaybeUninit<f64>>, Vec<f64>>(result_data)
    };

    // This might be a time-loss:
    let mut input_shape = [1usize;8];
    input_shape[0] = rows_a;
    input_shape[1] = cols_b;

    Ok(NdArray::new(result_data, &input_shape))

}


#[inline(always)]
fn offset(stride: &[usize;8], indices: [usize;2]) -> usize {
    stride[0] * indices[0] + stride[1] * indices[1]
}

#[inline(always)]
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

        let col_base = col * stride[0];
        let row_base = row * stride[0];

        if pivot_row != col {
            for k in 0..n {
                let col_k = offset(stride,[col,k]);
                let pivot_k = offset(stride,[pivot_row,k]);
                a.swap(col_k,pivot_k);
                inv_data.data.swap(col * n + k, pivot_row * n + k);
                
            }
        }

        let pivot_val = a[offset(stride,[col, col])];
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[offset(stride,[row, col])] / pivot_val;
            
            for k in 0..n {
                let col_k = col_base + stride[1] * k;
                let row_k = row_base + stride[1] * k;
                unsafe{
                    let a_val = *a.get_unchecked(col_k);
                    *a.get_unchecked_mut(row_k) -= factor * a_val;
                    let inv_val = *inv_data.data.get_unchecked(col * n + k);
                    *inv_data.data.get_unchecked_mut(row * n + k) -= factor * inv_val;
                }
            }
        }

        for k in 0..n {
            unsafe {
                let col_k = col_base + stride[1] * k;
                *a.get_unchecked_mut(col_k) /= pivot_val;
                *inv_data.data.get_unchecked_mut(col * n + k) /= pivot_val;
            }
        }
    }
    Ok(inv_data)
}