use crate::array::NdArray;

// This is vector to vector dot product
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

pub fn add(arr1: &NdArray, arr2: &NdArray) -> Result<NdArray,String> {
    // Matrices don't need to be equal size with the broadcasting function
    let (out_shape, strides1, strides2) = NdArray::broadcast(arr1, arr2)?;
    let total_elements = out_shape.iter().product();
    let mut index = vec![0;out_shape.len()];
    let mut data = Vec::with_capacity(total_elements);

    for _i in 0..total_elements {
        
        // Give an image in order to not consume RAM
        let val1 = NdArray::get_strides(arr1, &strides1, &index);
        let val2 = NdArray::get_strides(arr2, &strides2, &index);

        data.push(val1 + val2);


        for j in (0..out_shape.len()).rev() {
            index[j] += 1; // Increment this dimension

            if index[j] < out_shape[j] {
                // No carry needed, we are done updating for this step!
                break;
            } else {index[j] = 0;}
        }
    }
    Ok(NdArray::new(data, out_shape))
}

// Same logic as above function.
pub fn subtract(arr1: &NdArray, arr2: &NdArray) -> Result<NdArray,String> {
    // Matrices don't need to be equal size with the broadcasting function
    let (out_shape, strides1, strides2) = NdArray::broadcast(arr1, arr2)?;
    let total_elements = out_shape.iter().product();
    let mut index = vec![0;out_shape.len()];
    let mut data = Vec::with_capacity(total_elements);

    for _i in 0..total_elements {
        
        // Give an image in order to not consume RAM
        let val1 = NdArray::get_strides(arr1, &strides1, &index);
        let val2 = NdArray::get_strides(arr2, &strides2, &index);

        data.push(val1 - val2);


        for j in (0..out_shape.len()).rev() {
            index[j] += 1; // Increment this dimension

            if index[j] < out_shape[j] {
                // No carry needed, we are done updating for this step!
                break;
            } else {
                // Carry needed: reset this dimension and let the loop 
                // increment the next dimension (j-1)
                index[j] = 0;
            }
        }
    }
    Ok(NdArray::new(data, out_shape))
}

pub fn multiply(arr1: &NdArray, arr2: &NdArray) -> Result<NdArray,String> {
    // Matrices don't need to be equal size with the broadcasting function
    let (out_shape, strides1, strides2) = NdArray::broadcast(arr1, arr2)?;
    let total_elements = out_shape.iter().product();
    let mut index = vec![0;out_shape.len()];
    let mut data = Vec::with_capacity(total_elements);

    for _i in 0..total_elements {
        
        // Give an image in order to not consume RAM
        let val1 = NdArray::get_strides(arr1, &strides1, &index);
        let val2 = NdArray::get_strides(arr2, &strides2, &index);

        data.push(val1 * val2);


        for j in (0..out_shape.len()).rev() {
            index[j] += 1; // Increment this dimension

            if index[j] < out_shape[j] {
                // No carry needed, we are done updating for this step!
                break;
            } else {
                // Carry needed: reset this dimension and let the loop 
                // increment the next dimension (j-1)
                index[j] = 0;
            }
        }
    }
    Ok(NdArray::new(data, out_shape))
}

pub fn divide(arr1: &NdArray, arr2: &NdArray) -> Result<NdArray,String> {
    // Matrices don't need to be equal size with the broadcasting function
    let (out_shape, strides1, strides2) = NdArray::broadcast(arr1, arr2)?;
    let total_elements = out_shape.iter().product();
    let mut index = vec![0;out_shape.len()];
    let mut data = Vec::with_capacity(total_elements);

    for _i in 0..total_elements {
        
        // Give an image in order to not consume RAM
        let val1 = NdArray::get_strides(arr1, &strides1, &index);
        let val2 = NdArray::get_strides(arr2, &strides2, &index);

        data.push(val1 / val2);


        for j in (0..out_shape.len()).rev() {
            index[j] += 1; // Increment this dimension

            if index[j] < out_shape[j] {
                // No carry needed, we are done updating for this step!
                break;
            } else {
                // Carry needed: reset this dimension and let the loop 
                // increment the next dimension (j-1)
                index[j] = 0;
            }
        }
    }
    Ok(NdArray::new(data, out_shape))
}

// Add all values together
pub fn sum(arr: &NdArray) -> f64 {
    arr.data.iter().sum()
}

pub fn mean(arr: &NdArray) -> f64 {
    NdArray::sum(arr)/arr.data.len() as f64
}

pub fn max(arr: &NdArray) -> f64 {
    // Make sure max is a val in matrix
    let mut max = arr.data[0];
    for i in &arr.data{
        if *i > max{
            max = *i;
        }
    }
    max
}

pub fn min(arr: &NdArray) -> f64 {
    // Make sure max is a val in matrix
    let mut min = arr.data[0];
    for i in &arr.data{
        //Interestingly *i dereferences i
        if *i < min{
            min = *i;
        }
    }
    min
}