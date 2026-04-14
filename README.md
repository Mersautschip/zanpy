# 🚀 Zanpy

**Zanpy** is a high-performance N-dimensional array library built in **Rust** with a seamless **Python** interface. It leverages SIMD (Single Instruction, Multiple Data) and cache-efficient memory layouts to provide blazing-fast linear algebra operations.

---

## ✨ Features

* **Native Rust Engine:** Memory-safe, high-speed core implementation.
* **SIMD Acceleration:** Matrix multiplication optimized with `wide` SIMD types (f64x2) and $i, k, j$ loop reordering.
* **N-Dimensional Support:** Sophisticated broadcasting for arrays up to 8 dimensions.
* **Numpy-Compatible:** Familiar Python syntax for `reshape`, `transpose`, `arange`, and `identity`.
* **Linear Algebra:** Built-in LU decomposition for matrix inversion and dot product support.

---

## 📦 Installation

To install the latest release via `pip`:

```bash
pip install zanpy
