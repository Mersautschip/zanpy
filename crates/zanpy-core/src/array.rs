// Numpy NdArray is based on three things: data(raw), shape and strides. 
// Basically we are making a matrix into an array and then using modular arithmetic to calculate the matrix
// Shape needs to be inputed [x,y]
pub struct NdArray{
    // Data in the array will be stored in the f64 vector. 
    // (Data will be massive so storing in the heap doesn't matter)
    pub data: Vec<f64>,

    // The shape will be positive integers and isn't fixed from matrix to matrix
    pub shape: Vec<usize>,

    // Same goes for the stride
    pub stride: Vec<usize>,
}

impl NdArray {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> NdArray{
        // What I've found out is that the stride for the smallest dimension is always 1
        // Then the stride for the second smallest is the size of the smallest
        // This pattern follows geometrically

        // Since we'll have exactly as many strides as the shape
        let mut strides = vec![1; shape.len()];

        for i in (0..shape.len()-1).rev(){
            strides[i] = strides[i+1] * shape[i+1];
        }
        NdArray { data, shape, stride: strides}
        
    }

    // Function to get index in array of any matrix variable. 
    pub fn get(&self, indices: Vec<usize>) -> Result<f64, String> {
        // Standard foolproofing
        if indices.len() != self.stride.len() {
            return Err("Wrong number of indices".to_string());
        }
        Ok(self.data[self.offset(indices)])
    }
    
    pub fn get_strides(arr: &NdArray, stride: &[usize], indices: &[usize]) -> f64 {
        let mut offset: usize = 0;
        for i in 0..stride.len() {
            // This is the core of broadcasting: 
            // if stride[i] is 0, this dimension effectively disappears!
            offset += stride[i] * indices[i];
        }
        arr.data[offset]
    }

    fn offset(&self, indices: Vec<usize>) -> usize {
        //Initializing the offset
        let mut offset: usize = 0;

        //Basically applying the concept explained in new
        for i in 0..self.stride.len(){
            offset = offset + (self.stride[i]*indices[i])
        }
        offset
    }

    // Need to check if works
    fn rev_offset(&self, mut offset: usize) -> Vec<usize> {
        // Initialize the index vector
        let dimensions = self.shape.len();
        let mut indices = vec![0;dimensions];
        let mut skips = vec![1;dimensions];
        for i in (0..dimensions-1).rev(){
            skips[i] = skips[i+1] * skips[self.shape[i+1]]
        }
        for i in 0..dimensions {
            indices[i] = offset / skips[i];
            offset %= skips[i];
        }
        indices
    }

    pub fn set(&mut self, indices:Vec<usize>, val:f64) {
        self.data[self.offset(indices)] = val;
    } 

    // Creating a matrix filled with ones (Standard Numpy Function)
    pub fn ones(shape: Vec<usize>) -> NdArray{
        // Multiplying the values of the shape in order to find size
        let size= shape.iter().product();
        // Initializing data
        let data = vec![1.0; size];
        NdArray::new(data, shape)
    }

    // Now a function for 0
    pub fn zeros(shape: Vec<usize>) -> NdArray{
        // Multiplying the values of the shape in order to find size
        let size = shape.iter().product();
        // Initializing data
        let data = vec![0.0; size];
        NdArray::new(data, shape)
    }

    // Arange creates a 1d matrix with evenly spaced values. 
    // Start is the first value, end is the last, the step is the number of values
    // The function will never output the end value
    pub fn arange(start: f64, end: f64, step:f64) -> NdArray{
        // Vectors in Rust are dynamic arrays, with large values, dynamic arrays will
        // reorganize. And since we really only care about massive values, it is best 
        // to set the memory in order to assure O(1) speeds for assigning vars.
        let n = ((end-start)/step).ceil() as usize;
        let mut data = Vec::with_capacity(n);
        let mut curr = start;
        while curr < end{
            data.push(curr);
            curr += step;
        }

        NdArray::new(data, vec![1,n])
    }
    // This function will create an x by x identity matrix
    pub fn identity(x: usize) -> NdArray{
        let mut matrix = NdArray::zeros(vec![x,x]);
        // Not necessary to loop through, you already know the values
        for i in 0..x {
            matrix.set(vec![i, i], 1.0);
        }
        matrix
    }

    // Function to add equal matrices together
    pub fn reshape(arr: &NdArray, shape: Vec<usize>) -> Result<NdArray, String> {
        // Make sure that the new shape has an equal amount of values as original
        if arr.shape.iter().product::<usize>() != shape.iter().product::<usize>(){
            return Err("Shape values don't match".to_string())
        }
        let mut strides = vec![1; shape.len()];

        for i in (0..shape.len()-1).rev(){
            strides[i] = strides[i+1] * shape[i+1];
        }
        Ok(NdArray { data: arr.data.clone(), shape , stride: strides})
    } 

    pub fn transpose(arr: &NdArray) -> NdArray {
        let mut shape = arr.shape.clone();
        shape.reverse();
        let mut stride = arr.stride.clone();
        stride.reverse();
        NdArray { data: arr.data.clone(), shape, stride }
    }

    pub fn broadcast(arr1: &NdArray, arr2: &NdArray) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>),String> {
        let len1 = arr1.shape.len();
        let len2 = arr2.shape.len();
        let maxlen = len1.max(len2);
        let mut newdim = vec![1;maxlen];
        let mut strides1 = vec![1;maxlen];
        let mut strides2 = vec![1;maxlen];

        for i in 0..maxlen{
            let dim1 = if i < len1 {arr1.shape[len1-i-1]} else {1};
            let dim2 = if i < len2 {arr2.shape[len2-i-1]} else {1};

            if dim1 != dim2 && dim1 != 1 && dim2 != 1{
                return Err("Dimensions are not compatible".to_string());
            }
            else{
                newdim[i] = dim1.max(dim2);
                strides1[i] = if dim1 == 1 { 0 } else { arr1.stride[len1 - i - 1] };
                strides2[i] = if dim2 == 1 { 0 } else { arr2.stride[len2 - i - 1] };
            }
        }
        newdim.reverse();
        strides1.reverse();
        strides2.reverse();
        Ok((newdim, strides1, strides2))

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let arr = NdArray::new(data, shape);
        assert_eq!(arr.stride, vec![3, 1]);
        assert_eq!(arr.shape, vec![2, 3]);
    }

    #[test]
    fn test_get() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let arr = NdArray::new(data, shape);
        assert_eq!(arr.get(vec![0, 0]).unwrap(), 1.0);
        assert_eq!(arr.get(vec![0, 1]).unwrap(), 2.0);
        assert_eq!(arr.get(vec![0, 2]).unwrap(), 3.0);
        assert_eq!(arr.get(vec![1, 0]).unwrap(), 4.0);
        assert_eq!(arr.get(vec![1, 1]).unwrap(), 5.0);
        assert_eq!(arr.get(vec![1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_set() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let mut arr = NdArray::new(data, shape);
        arr.set(vec![0, 0], 99.0);
        assert_eq!(arr.get(vec![0, 0]).unwrap(), 99.0);
        arr.set(vec![1, 2], 42.0);
        assert_eq!(arr.get(vec![1, 2]).unwrap(), 42.0);
    }

    #[test]
    fn test_zeros() {
        let arr = NdArray::zeros(vec![2, 3]);
        assert_eq!(arr.get(vec![0, 0]).unwrap(), 0.0);
        assert_eq!(arr.get(vec![1, 2]).unwrap(), 0.0);
        assert_eq!(arr.shape, vec![2, 3]);
    }

    #[test]
    fn test_ones() {
        let arr = NdArray::ones(vec![2, 3]);
        assert_eq!(arr.get(vec![0, 0]).unwrap(), 1.0);
        assert_eq!(arr.get(vec![1, 2]).unwrap(), 1.0);
        assert_eq!(arr.shape, vec![2, 3]);
    }

    #[test]
    fn test_arange() {
        let arr = NdArray::arange(0.0, 6.0, 1.0);
        assert_eq!(arr.get(vec![0, 0]).unwrap(), 0.0);
        assert_eq!(arr.get(vec![0, 1]).unwrap(), 1.0);
        assert_eq!(arr.get(vec![0, 5]).unwrap(), 5.0);
        let arr2 = NdArray::arange(0.0, 10.0, 2.0);
        assert_eq!(arr2.get(vec![0, 0]).unwrap(), 0.0);
        assert_eq!(arr2.get(vec![0, 2]).unwrap(), 4.0);
        assert_eq!(arr2.get(vec![0, 4]).unwrap(), 8.0);
    }

    #[test]
    fn test_identity() {
        let arr = NdArray::identity(3);
        assert_eq!(arr.get(vec![0, 0]).unwrap(), 1.0);
        assert_eq!(arr.get(vec![1, 1]).unwrap(), 1.0);
        assert_eq!(arr.get(vec![2, 2]).unwrap(), 1.0);
        assert_eq!(arr.get(vec![0, 1]).unwrap(), 0.0);
        assert_eq!(arr.get(vec![1, 0]).unwrap(), 0.0);
        assert_eq!(arr.get(vec![0, 2]).unwrap(), 0.0);
    }
}