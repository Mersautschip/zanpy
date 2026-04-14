// Numpy NdArray is based on three things: data(raw), shape and strides. 
// Basically we are making a matrix into an array and then using modular arithmetic to calculate the matrix
// Shape needs to be inputed [x,y]
// I'm going to maximize the dimensions to 8. This is a speed/availability trade-off 
// As a vec will lie on the heap and not the stack. Realistically no-one will surpass 8 dimensions
pub struct NdArray{
    // Data in the array will be stored in the f64 vector. 
    // (Data will be massive so storing in the heap doesn't matter)
    pub data: Vec<f64>,

    // The shape will be positive integers and isn't fixed from matrix to matrix
    pub shape: [usize;8],
    // Shape will be read left to right in Col Major ordering

    // Same goes for the stride
    pub stride: [usize;8],

    pub rank: usize, // This tells the program the dimensions of the matrix
}

impl NdArray {
    pub fn new(data: Vec<f64>, input_shape: &[usize]) -> NdArray{
        // What I've found out is that the stride for the smallest dimension is always 1
        // Then the stride for the second smallest is the size of the smallest
        // This pattern follows geometrically
        

        let rank = input_shape.len();
        assert!(rank<=8, "zanpy only supports up to 8 dimensions!");
        
        let mut shape = [0;8];
        let mut stride = [1; 8];

        for i in 0..rank{
            shape[i] = input_shape[i];
        }

        for i in (0..rank-1).rev(){
            stride[i] = stride[i+1] * shape[i+1];
        }
        NdArray { data, shape, stride, rank}
        
    }

    // Function to get index in array of any matrix variable. 
    pub fn get(&self, indices: &[usize]) -> Result<f64, String> {
        // Standard foolproofing
        if indices.len() != self.rank {
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

    #[inline(always)]
    fn offset(&self, indices: &[usize]) -> usize {
        //Initializing the offset
        let rank = self.rank;
        assert!(indices.len()== rank, "Missing dimensions");
        let mut offset: usize = 0;
        
        unsafe{
            let stride_ptr = self.stride.as_ptr();
            let indices_ptr = indices.as_ptr();

            for i in 0..rank{
                let s = *stride_ptr.add(i);
                let idx = *indices_ptr.add(i);
                offset += s * idx;
            }
        }

        offset
    }

    // Need to check if works
    // Placeholder function tbh
    #[allow(dead_code)]
    fn rev_offset(&self, mut offset: usize) -> [usize;8] {
        // Initialize the index vector
        let dimensions = self.shape.len();
        let mut indices = [0;8];
        let mut skips = [1;8];
        for i in (0..dimensions-1).rev(){
            skips[i] = skips[i+1] * skips[self.shape[i+1]]
        }
        for i in 0..dimensions {
            indices[i] = offset / skips[i];
            offset %= skips[i];
        }
        indices
    }

    pub fn set(&mut self, indices:&[usize], val:f64) {
        let idx = self.offset(indices);
        self.data[idx] = val;
    } 

    // Creating a matrix filled with ones (Standard Numpy Function)
    pub fn ones(shape: &[usize]) -> NdArray{
        // Multiplying the values of the shape in order to find size
        let size= shape.iter().product();
        // Initializing data
        let data = vec![1.0; size];
        NdArray::new(data, shape)
    }

    // Now a function for 0
    pub fn zeros(shape: &[usize]) -> NdArray{
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
        // Vectors in Rust are dynamic arrays. With large values, dynamic arrays will
        // reorganize. And since we really only care about massive values, it is best 
        // to set the memory in order to assure O(1) speeds for assigning vars.
        let n = ((end-start)/step).ceil() as usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            data.push(start + (i as f64 * step));
        }

        NdArray::new(data, &[n])
    }
    // This function will create an x by x identity matrix
    pub fn identity(x: usize) -> NdArray{
        let mut matrix = NdArray::zeros(&[x,x]);
        // Not necessary to loop through, you already know the values
        for i in 0..x {
            matrix.set(&[i, i], 1.0);
        }
        matrix
    }

    // Function to add equal matrices together
    pub fn reshape(arr: &NdArray, shape: &[usize]) -> Result<NdArray, String> {
        // Make sure that the new shape has an equal amount of values as original
        if arr.shape.iter().product::<usize>() != shape.iter().product::<usize>(){
            return Err("Shape values don't match".to_string())
        }
        let mut strides = [1; 8];
        let mut new_shape = [0; 8];
        let rank = shape.len();
        for i in 0..rank{
            new_shape[i] = shape[i];
        }

        for i in (0..rank-1).rev(){
            strides[i] = strides[i+1] * shape[i+1];
        }
        Ok(NdArray { data: arr.data.clone(), shape:new_shape , stride: strides, rank})
    } 

    pub fn broadcast(arr1: &NdArray, arr2: &NdArray) -> Result<([usize;8], [usize;8], [usize;8]),String> {
        let len1 = arr1.rank;
        let len2 = arr2.rank;
        const MAXLEN:usize = 8;
        let mut newdim = [1;MAXLEN];
        let mut strides1 = [1;MAXLEN];
        let mut strides2 = [1;MAXLEN];

        if arr1.shape == arr2.shape{
            return Ok((arr1.shape, arr1.stride,arr2.stride));
        }

        for i in 0..MAXLEN{
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

    pub fn transpose(&mut self) -> Result<(), String> {
        if self.shape.len() != 2 {
            return Err("Transposition is only allowed for two dimensions".to_string())
        }
        self.shape.reverse();
        self.stride.reverse();
        Ok(())
    }

    // Transposition in multiple dimensions(works for 2d as well)
    pub fn permute(&self, axes: &[usize]) -> Result<NdArray, String> {
        let rank = axes.len();
        if rank != self.shape.len() { 
            return Err("Permutation must match the number of dimensions".to_string());
        }

        let mut new_shape = [0; 8];
        let mut new_stride = [0; 8];

        for (i, &axis_idx) in axes.iter().enumerate() {
            new_shape[i] = self.shape[axis_idx];
            new_stride[i] = self.stride[axis_idx];
        }

        Ok(NdArray {
            data: self.data.clone(), // Still zero-copy if using Rc
            shape: new_shape,
            stride: new_stride,
            rank,
        })
    }
}