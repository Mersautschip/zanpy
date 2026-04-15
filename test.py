import numpy as np
import time
import zanpy  # Ensure you have run 'maturin develop'
import sys

def run_benchmark(size, iterations=10):
    print(f"\n🚀 Benchmarking Matrix Size: {size}x{size} ({iterations} iterations)")
    print("-" * 50)

    # 1. Setup Data
    # We use random data but ensure it's f64 (float64) to match your Rust Vec<f64>
    a_raw = np.random.randn(size, size).astype(np.float64)
    b_raw = np.random.randn(size, size).astype(np.float64)

    # Pre-convert to Zanpy objects so we only measure the math, not the allocation
    z_a = zanpy.PyNdArray(a_raw.flatten().tolist(), [size, size])
    z_b = zanpy.PyNdArray(b_raw.flatten().tolist(), [size, size])

    # 2. Benchmark Numpy
    # Numpy uses OpenBLAS/MKL which is highly optimized for the M1
    np_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = a_raw @ b_raw
        np_times.append(time.perf_counter() - start)
    
    avg_np = sum(np_times) / iterations
    print(f"Numpy Avg: {avg_np:.6f}s")

    # 3. Benchmark Zanpy
    # This triggers your __matmul__ magic method and f64x2 SIMD logic
    zp_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = z_a @ z_b
        zp_times.append(time.perf_counter() - start)
    
    avg_zp = sum(zp_times) / iterations
    print(f"Zanpy Avg: {avg_zp:.6f}s")

    # 4. Results
    ratio = avg_zp / avg_np
    print(f"Result: Zanpy is {ratio:.2f}x {'faster' if ratio < 1 else 'slower'} than Numpy")

    # 5. Correctness Check
    # Let's make sure your SIMD math actually produces the right numbers
    z_res = z_a @ z_b
    # Accessing the data back (requires a .data() getter in your Rust code)
    try:
        z_res_np = np.array(z_res.data).reshape(size, size)
        np.testing.assert_allclose(a_raw @ b_raw, z_res_np, rtol=1e-7)
        print("✅ Math Verification: PASSED")
    except AttributeError:
        print("⚠️  Could not verify math: Add a '.data' getter to PyNdArray")
    except AssertionError as e:
        print(f"❌ Math Verification: FAILED\n{e}")

if __name__ == "__main__":
    # Test different scales
    # 128: Fits in L1/L2 cache
    # 512: Pushes L3 cache
    # 1024: Memory bandwidth bound
    sizes = [128, 512, 1024]
    
    for s in sizes:
        run_benchmark(s, iterations=5 if s > 512 else 20)