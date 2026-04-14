import zanpy as zp

# Initialize your custom NdArray
# This calls your Rust NdArray::new(data, &shape)
arr = zp.PyNdArray([10.0, 20.0, 30.0, 40.0], [2, 2])

# Access an element
# This calls your Rust self.inner.get(&indices)
val = arr.get([1, 0])

print(f"Value at [1, 0]: {val}")
print(f"Type of object: {type(arr)}")