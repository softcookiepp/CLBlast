# CLBlast-vk – Agent Guide

## Project Overview

**CLBlast-vk** is a fork of [CLBlast](https://github.com/CNugteren/CLBlast) that replaces its OpenCL backend with a custom Vulkan compute engine called **Tart**. CLBlast is a tuned OpenCL/BLAS library; this fork produces a Vulkan-based BLAS library (`libclblast`) that targets Vulkan compute shaders written in GLSL.

The project is **not** intended to be merged back into upstream CLBlast. It is a standalone fork that will eventually be renamed or factored into another library.

### The Tart Subproject

Tart (`tart/`) is a Vulkan-based GPGPU compute framework written in C++17. It wraps Vulkan's low-level compute API into a higher-level, less-boilerplate experience while preserving flexibility. Tart provides:

- A C++ API (`tart::Instance`, `tart::Device`, `tart::Buffer`, `tart::Pipeline`, `tart::Kernel`, etc.)
- Python bindings (`pytart`) via pybind11 + scikit-build-core
- GLSL (via glslang) and OpenCL C (via external clspv) shader compilation
- Vulkan Memory Allocator (VMA) integration for GPU memory management
- SPIR-V reflection for automatic descriptor binding and push constant alignment

Tart lives as a **git submodule** at `tart/` pointing to `https://codeberg.org/softcookiepp/tart.git`. Its own documentation is in `tart/AGENTS.md`.

### Licensing

- **CLBlast root**: Apache License 2.0 (copyright Cedric Nugteren, 2015)
- **Tart subproject**: MIT License (copyright Thales Sabino)

---

## Repository Structure

```
CLBlast-vk/                          # Root: CLBlast fork, Vulkan BLAS library
├── include/                         # Public headers (user-facing API)
│   ├── clblast_vk.h                 # Main header: BLAS routine declarations (Vulkan)
│   └── clblast_half.h               # Half-precision helper
├── src/                             # CLBlast library source
│   ├── clblast_vk.cpp               # Vulkan-specific BLAS API implementation
│   ├── vkpp11.cpp / vkpp11.hpp     # Vulkan C++11 wrapper (OpenCL-like API over Tart)
│   ├── routine.hpp / .cpp          # Base class for all BLAS routines
│   ├── routine_aot_cache.hpp       # AOT kernel getter functions (precompiled SPIR-V)
│   ├── cache.hpp / .cpp            # Kernel compile-time cache
│   ├── database/                   # Database layer for device-specific tuning params
│   │   ├── database.cpp            # Database initialization and lookup
│   │   ├── database_structure.hpp  # Database data structures (NewDatabaseEntry, etc.)
│   │   └── kernels/                # Database kernel definitions (copy, gemm_routine, etc.)
│   ├── routines/                   # BLAS routine implementations
│   │   ├── level1/                 # Level-1 (vector-vector) routines
│   │   ├── level2/                 # Level-2 (matrix-vector) routines
│   │   ├── level3/                 # Level-3 (matrix-matrix) routines
│   │   ├── levelx/                 # Extra non-BLAS routines (convolution, etc.)
│   │   └── common.cpp / .hpp       # Shared routine utilities
│   ├── tuning/                     # Auto-tuning infrastructure
│   │   ├── tuning.cpp / .hpp       # Tuning engine
│   │   ├── tuning_api.cpp          # Public tuning API (RetrieveParameters, OverrideParameters)
│   │   ├── configurations.cpp      # Kernel parameter configurations
│   │   ├── kernels/                # Per-kernel tuners (xaxpy, xgemm, etc.)
│   │   └── routines/               # Per-routine tuners (xgemm, xtrsv)
│   ├── utilities/                  # Shared utilities
│   │   ├── backend.hpp             # Backend selector (OPENCL_API vs VULKAN_API)
│   │   ├── compile.cpp / .hpp      # Shader compilation helpers
│   │   ├── timing.cpp / .hpp       # Performance timing utilities
│   │   ├── utilities.cpp / .hpp    # General utilities
│   │   └── clblast_exceptions.hpp  # Exception types
│   ├── kernels-vk/                 # GLSL shader sources for Vulkan kernels
│   │   ├── common.glsl             # Shared GLSL utilities
│   │   ├── dtype-enum.glsl         # Data type enumeration
│   │   ├── level1/                 # Level-1 compute kernels
│   │   ├── level2/                 # Level-2 compute kernels
│   │   ├── level3/                 # Level-3 compute kernels
│   │   └── levelx/                 # Extra kernels (im2col, convgemm, etc.)
│   └── kernels-vk-inline/          # Generated C++ headers for inline SPIR-V
├── tart/                            # Tart subproject (Vulkan GPGPU framework)
│   ├── include/tart/                # Public headers (tart.hpp, dtypes.hpp, etc.)
│   ├── src/                         # Tart implementation
│   ├── tests/                       # Doctest-based test suite
│   ├── 3rdparty/                    # Third-party dependencies (git submodules)
│   ├── CMakeLists.txt               # Tart build system (CMake >= 3.20)
│   └── pyproject.toml               # Python bindings config
├── test/                            # CLBlast test suite
│   ├── correctness/                 # Correctness tests vs reference BLAS
│   ├── performance/                 # Performance benchmark clients
│   ├── routines/                    # Test routine headers
│   └── wrapper_*.hpp                # Reference library wrappers (CBLAS, cuBLAS)
├── samples/                         # Example programs
├── scripts/                         # Build and utility scripts
│   ├── inline_gen.py               # Generates inline SPIR-V headers
│   ├── benchmark/                  # Benchmark scripts
│   └── database/                   # Database management scripts
├── cmake/                           # CMake modules
│   ├── compile_kernels.cmake       # Kernel source generation logic
│   └── Modules/                    # Find modules (CBLAS, MKL)
├── doc/                             # Documentation
├── CMakeLists.txt                   # Root build system (CMake >= 3.10)
├── vcpkg.json                       # vcpkg manifest (no external deps)
├── .gitmodules                      # Tart submodule definition
└── .github/workflows/               # CI pipelines
```

---

## Architecture

### High-Level Design

```
User Application
     │
     ▼
clblast::Gemm() / clblast::Axpy() / ...   (template API, clblast_vk.h)
     │
     ▼
Routine<T> base class                      (compiles shaders, connects to database)
     │
     ├── tart::Device (via vkpp11 wrapper)
     │   ├── tart::Buffer (GPU memory via VMA)
     │   ├── tart::Pipeline (Vulkan compute pipeline)
     │   └── tart::Kernel (OpenCL-style kernel wrapper)
     │
     ├── Database                         (device-specific tuning parameters)
     │   └── NewDatabaseEntry per kernel/precision/device
     │
     └── AOT Kernel Cache                 (precompiled SPIR-V via tart-spirv-inliner)
```

### Key Components

#### 1. Backend Selection (CMake)

The `VULKAN` CMake option (default ON) selects the Vulkan code path:
- Defines `VULKAN_API` preprocessor macro
- Includes `clblast_vk.h` as the main header
- Compiles `clblast_vk.cpp` as the implementation
- Links against the Tart library

The `VULKAN` option (always ON) selects the Vulkan code path:
- Defines `VULKAN_API`
- Includes `clblast_vk.h` as the main header
- Compiles `clblast_vk.cpp` as the implementation
- Links against the Tart library

The `OPENCL` option is deprecated (no longer builds).

#### 2. vkpp11 – Vulkan C++11 Wrapper

`src/vkpp11.hpp` provides an OpenCL-like C++ API that wraps Tart:
- `clblast::Device` wraps `tart::device_ptr`
- `clblast::Buffer<T>` wraps `tart::buffer_ptr`
- `clblast::Program` wraps Tart's multi-shader compilation
- `clblast::Queue` wraps Tart's command queue
- `clblast::Kernel` wraps `tart::kernel_ptr` with argument setting

This allows the CLBlast routine code to remain largely unchanged between backends.

#### 3. Database Layer

The database system stores device-specific tuning parameters for each kernel/precision combination:

- **`NewDatabaseEntry`**: Contains kernel name, precision, definition names, specialization constant names, and vendor-specific parameters
- **`NewDatabaseDevice`**: Per-device entries with device ID, vendor ID, and parameter arrays
- **`Databases`**: The runtime-lookup structure built from the database

At runtime, the library:
1. Queries the device for its capabilities
2. Looks up the best parameters from the database (or falls back to defaults)
3. Applies define-based and specialization constant parameters when compiling shaders

#### 4. AOT Kernel Cache

`src/routine_aot_cache.hpp` provides precompiled kernel access functions:

```cpp
// Get a precompiled kernel for a specific routine, precision, and vector width
tart::kernel_ptr getAxpy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);
tart::kernel_ptr getGemm(const tart::device_ptr& device, const tart::DType& dtype, ...);
```

These return kernels compiled from SPIR-V binaries generated by the `tart-spirv-inliner` tool during the build. The kernel sources in `src/kernels-vk-inline/` are auto-generated from `.spv` files.

#### 5. Shader Compilation Pipeline

1. **GLSL source files** in `src/kernels-vk/` define the compute kernels
2. **Build-time**: `tart-spirv-inliner` compiles GLSL → SPIR-V and generates C++ source with embedded SPIR-V constants
3. **Runtime**: Tart's `Device::compileGLSL()` compiles GLSL → SPIR-V (via linked glslang or external `glslc` subprocess)
4. **Runtime**: Tart's SPIR-V reflection (via SPIRV-Reflect) extracts descriptor bindings and push constant layouts automatically

#### 6. Tuning System

The auto-tuner discovers optimal kernel parameters for each GPU:

- **Kernel tuners** (`src/tuning/kernels/`): Tune individual kernel parameters (xaxpy, xgemm, xdot, etc.)
- **Routine tuners** (`src/tuning/routines/`): Tune entire routines (xgemm, xtrsv)
- **Tuning API** (`src/tuning/tuning_api.cpp`): Public functions `RetrieveParameters()` and `OverrideParameters()`
- **Build-time**: `clblast_tuner_<kernel>` executables are generated for each kernel

The `alltuners` CMake target runs all tuners for all precisions.

#### 7. Tart's PIMPL Architecture

Tart uses an **Implicit PIMPL** pattern where internal infrastructure classes (DescriptorSetCache, PipelineLayoutCache, etc.) are `std::unique_ptr` members of `Device`, declared as friends. Public header `tart.hpp` contains only forward declarations. See `tart/AGENTS.md` for details.

---

## BLAS Routines Supported

The library implements the following BLAS routines, organized by level:

### Level-1 (Vector-Vector)

| Routine | Functions | GLSL Kernels |
|---------|-----------|-------------|
| XSWAP | Swap two vectors | `xswap.glsl`, `xswap_fast.glsl` |
| XSCAL | Vector scaling | `xscal.glsl`, `xscal_fast.glsl` |
| XCOPY | Vector copy | `xcopy.glsl`, `xcopy_fast.glsl` |
| XAXPY | y = alpha*x + y | `xaxpy.glsl`, `xaxpy_faster.glsl`, `xaxpy_fastest.glsl` |
| XDOT | Dot product | `xdot.glsl`, `xdot_epilogue.glsl` |
| XDOTU | Complex dot product (no conjugate) | `xdot.glsl` |
| XDOTC | Complex dot product (conjugate) | `xdot.glsl` |
| XNRM2 | Euclidean norm | `xnrm2.glsl`, `xnrm2_epilogue.glsl` |
| XASUM | Absolute sum | `xasum.glsl`, `xasum_epilogue.glsl` |
| XAMAX | Index of absolute max | `xamax.glsl`, `xamax_epilogue.glsl` |
| XAMIN | Index of absolute min | (inline in level1 routines) |
| XMAX | Index of max | (inline) |
| XMIN | Index of min | (inline) |
| XSUM | Sum (non-BLAS) | (inline) |
| XROTG/XROTMG | Givens plane rotation | (inline) |
| XROT/XROTM | Apply Givens rotation | (inline) |

### Level-2 (Matrix-Vector)

| Routine | Functions | GLSL Kernels |
|---------|-----------|-------------|
| XGEMV | General matrix-vector | `xgemv.glsl`, `xgemv_fast.glsl`, `xgemv_fast_rot.glsl` |
| XGBMV | Banded general matrix-vector | (via XGEMV) |
| XHEMV | Hermitian matrix-vector | (via XGEMV) |
| XHBMV | Hermitian banded | (via XGEMV) |
| XHPMV | Hermitian packed | (via XGEMV) |
| XSYMV | Symmetric matrix-vector | (via XGEMV) |
| XSBMV | Symmetric banded | (via XGEMV) |
| XSPMV | Symmetric packed | (via XGEMV) |
| XTRMV | Triangular matrix-vector | (via XGEMV) |
| XTBMV | Triangular banded | (via XGEMV) |
| XTPMV | Triangular packed | (via XGEMV) |
| XTRSV | Triangular solve | `xtrsv_forward.glsl`, `xtrsv_backward.glsl`, `xtrsv_fill_vector.glsl` |
| XTBSV | Triangular banded solve | (via XTRSV) |
| XTPSV | Triangular packed solve | (via XTRSV) |
| XGER | General rank-1 update | `xger.glsl` |
| XGERU | Complex rank-1 update | `xger.glsl` |
| XGERC | Complex rank-1 conjugate update | `xger.glsl` |
| XHER | Hermitian rank-1 update | `xher.glsl` |
| XHPR | Hermitian packed rank-1 | `xher.glsl` |
| XHER2 | Hermitian rank-2 update | `xher2.glsl` |
| XHPR2 | Hermitian packed rank-2 | `xher2.glsl` |
| XSYR | Symmetric rank-1 update | (via XHER) |
| XSPR | Symmetric packed rank-1 | (via XHER) |
| XSYR2 | Symmetric rank-2 update | (via XHER2) |
| XSPR2 | Symmetric packed rank-2 | (via XHER2) |

### Level-3 (Matrix-Matrix)

| Routine | Functions | GLSL Kernels |
|---------|-----------|-------------|
| XGEMM | General matrix-matrix multiply | `xgemm.glsl`, `xgemm_part1..4.glsl`, `xgemm_upper/lower.glsl` |
| XGEMM_DIRECT | Direct GEMM (unrolled) | `xgemm_direct.glsl`, `xgemm_direct_part1..3.glsl` |
| XSYMM | Symmetric matrix-matrix | (via XGEMM) |
| XHEMM | Hermitian matrix-matrix | (via XGEMM) |
| XSYRK | Symmetric rank-K update | (via XGEMM) |
| XHERK | Hermitian rank-K update | (via XGEMM) |
| XSYR2K | Symmetric rank-2K update | (via XGEMM) |
| XHER2K | Hermitian rank-2K update | (via XGEMM) |
| XTRMM | Triangular matrix-matrix | (via XGEMM) |
| XTRSM | Triangular solve (matrix) | `invert_diagonal_block.glsl`, `invert_diagonal_blocks_part1/2.glsl` |

### Level-X (Extra Non-BLAS)

| Routine | Functions | GLSL Kernels |
|---------|-----------|-------------|
| XHAD | Element-wise Hadamard product | `xhad.glsl`, `xhad_faster.glsl`, `xhad_fastest.glsl` |
| XOMATCOPY | Out-of-place matrix transpose/copy | (via copy/transpose kernels) |
| XIM2COL | Image-to-col (for convolutions) | `im2col_kernel_normal.glsl`, `im2col_col2im_common.glsl` |
| XCOL2IM | Col-to-image | `col2im_kernel_normal.glsl` |
| XCONVGEMM | Batched convolution as GEMM | `convgemm_kernel.glsl`, `convgemm_normal.glsl`, `xconvgemm_part1/2.glsl` |
| XAXPYBATCHED | Batched AXPY | `xaxpy_batched.glsl` |
| XGEMMBATCHED | Batched GEMM | `xgemm_batched.glsl` |
| XGEMMSTRIDEDBATCHED | Strided-batched GEMM | `xgemm_strided_batched.glsl` |
| XINVERT | Invert small matrices | `invert_diagonal_blocks_part2_*/glsl` |

### Supported Precisions

| Precision | DType | C++ Type |
|-----------|-------|----------|
| kHalf (16) | DTYPE_F16 | `half` |
| kSingle (32) | DTYPE_F32 | `float` |
| kDouble (64) | DTYPE_F64 | `double` |
| kComplexSingle (3232) | DTYPE_C32 | `std::complex<float>` |
| kComplexDouble (6464) | DTYPE_C64 | `std::complex<double>` |

---

## Vulkan Kernels (GLSL Shader Sources)

All compute kernels are written in GLSL `#version 450` (compute shaders) and organized by BLAS level:

### Common Shader Utilities

- `src/kernels-vk/common.glsl` – Shared helper functions
- `src/kernels-vk/dtype-enum.glsl` – Data type enumeration and conversion

### Level-1 Kernels (`src/kernels-vk/level1/`)

Vector operations: AXPY, COPY, SCAL, SWAP, DOT, NRM2, ASUM, AMAX, HAD (element-wise multiply).

### Level-2 Kernels (`src/kernels-vk/level2/`)

Matrix-vector operations: GEMV, GER, HER, HER2, TRSV (forward/backward/fill-vector).

### Level-3 Kernels (`src/kernels-vk/level3/`)

Matrix-matrix operations: GEMM (multi-part unrolled implementation), GEMM_DIRECT (direct unrolled), TRANSPOSE (fast/pad variants), COPY (fast/pad variants), INVERT (diagonal block inversion for TRSM), and matrix conversion helpers (Hermitian, symmetric, triangular).

### Level-X Kernels (`src/kernels-vk/levelx/`)

Extra operations: IM2COL, COL2IM, CONVGEMM (convolution as GEMM with im2col and normal variants).

### Batched and Strided-Batched Variants

Many Level-3 kernels have `_batched` and `_strided_batched` variants for batched operations.

---

## Build System

### Root Project (CLBlast-vk)

**Requirements**: CMake >= 3.10, C++17 compiler, Tart submodule

```bash
git clone --recurse-submodules https://<repo>/CLBlast-vk.git
cd CLBlast-vk
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

#### Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `VULKAN` | ON | Build with Vulkan backend |
| `OPENCL` | OFF | Deprecated (no longer builds) |
| `VULKAN_BDA` | OFF | Enable Vulkan Buffer Device Address support |
| `SAMPLES` | OFF | Build sample programs |
| `TUNERS` | ON | Build tuning executables |
| `CLIENTS` | OFF | Build performance benchmark clients |
| `TESTS` | OFF | Build correctness tests |
| `CUBLAS` | OFF | Deprecated (cuBLAS wrapper removed) |
| `NETLIB` | OFF | Deprecated (netlib API removed) |
| `VERBOSE` | OFF | Enable diagnostic messages |
| `BUILD_SHARED_LIBS` | ON | Build shared library (OFF for static) |

#### Build Targets

- `clblast` – The main shared library (`libclblast.so`)
- `alltuners` – Builds and runs all kernel tuners
- `alltests` – Builds and runs all correctness tests
- `clblast_tuner_<kernel>` – Individual kernel tuners
- `clblast_client_<routine>` – Performance benchmark clients
- `clblast_test_<routine>` – Correctness tests
- `clblast_sample_*` – Example programs

### Tart Subproject

**Requirements**: CMake >= 3.20, Vulkan SDK, C++17

See `tart/CMakeLists.txt` for full details. Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `TART_ENABLE_SHADER_COMPILERS` | ON | Embed glslang compiler |
| `TART_USE_SYSTEM_GLSLANG` | ON | Use system glslang vs. embedded |
| `TART_ENABLE_PYBIND` | OFF | Build Python bindings |
| `TART_ENABLE_TESTS` | OFF | Build Tart test suite |
| `TART_ENABLE_PROFILING` | OFF | Enable host profiling |
| `TART_ENABLE_VALIDATION_LAYERS` | OFF | Enable Vulkan validation layers |
| `TART_BUILD_STATIC` | OFF | Build static library |
| `TART_BUILD_NO_SYMLINKS` | OFF | Skip soname symlinks (for packaging) |

#### Tart Build Artifacts

- `libtart.so.0.1.0` – Shared library with soname
- `run-tests` – Tart test executable (when `TART_ENABLE_TESTS=ON`)
- `pytart` – Python bindings (when `TART_ENABLE_PYBIND=ON`)
- `tart-spirv-inliner` – SPIR-V compiler tool (built as subdirectory)

#### Kernel Source Generation (Tart)

The `tart_add_kernel_set()` CMake macro compiles GLSL → SPIR-V → C++ source files:

```cmake
tart_add_kernel_set(
    NAME kernels
    INPUTS
        test_kernels ${PROJECT_SOURCE_DIR}/tests/axpby.glsl
    COMPILER_ARGS
        -DDTYPE=DTYPE_F32 -DVW=1
)
```

This requires `tart-spirv-inliner` to be built first. The generated `.hpp`/`.cpp` files contain embedded SPIR-V as C++ arrays.

---

## Testing

### CLBlast Correctness Tests

Located in `test/correctness/`. Each routine has a test that compares results against a reference BLAS library (OpenBLAS, MKL, or cuBLAS).

```bash
cmake .. -DTESTS=ON -DCBLAS_ROOT=...
make alltests
ctest --test-dir build
```

### CLBlast Performance Tests

Located in `test/performance/`. Benchmark clients compare CLBlast-vk performance against reference BLAS.

```bash
cmake .. -DCLIENTS=ON -DCBLAS_ROOT=...
make clblast_client_xgemm
./build/clblast_client_xgemm
```

### Tart Unit Tests

Located in `tart/tests/`. Uses the **doctest** framework (git submodule). Each test is a `.hpp` file with `TEST_CASE()` blocks.

```bash
cd tart
mkdir build && cd build
cmake .. -DTART_ENABLE_TESTS=ON
make
./run-tests
```

Tart tests cover:
- Memory allocation and deallocation
- Vulkan device enumeration and capabilities
- Shader compilation (GLSL and OpenCL C via clspv)
- Kernel execution with various data types (f16, f32, f64, i32, u32, complex)
- Push constant behavior, descriptor set management
- Subgroup operations, atomics
- Buffer Device Address (BDA) support
- Semaphores, command buffers, profiling
- Mappable (pinned) memory I/O
- Error handling and OOM recovery

### Adding CLBlast Tests

1. Create `test/correctness/routines/level<N>/<routine>.cpp`
2. Create `test/routines/level<N>/<routine>.hpp`
3. Add to `CMakeLists.txt` routine lists (LEVEL1_ROUTINES, LEVEL2_ROUTINES, etc.)
4. Add to `src/routines/` with corresponding `.cpp` and `.hpp` files

### Adding Tart Tests

1. Create `tart/tests/test-<feature>.hpp`
2. Include `tests-common.hpp` and `tart.hpp`
3. Write `TEST_CASE()` blocks
4. Add `#include "test-<feature>.hpp"` to `tart/tests/tests.cpp`

---

## Tuning System

### How It Works

1. **Database entries** in `src/database/kernels/` define parameter names and default values per vendor/device
2. **Tuner executables** (`clblast_tuner_<kernel>`) run performance benchmarks across parameter combinations
3. **Best parameters** are stored in the database and loaded at runtime
4. **OverrideParameters()** allows users to manually set parameters

### Tuner Kernels (`src/tuning/kernels/`)

| Tuner | Routines Affected |
|-------|-------------------|
| xaxpy | Level-1 vector ops (AXPY, COPY, SWAP, SCAL) |
| xdot | Level-1 reduction ops (DOT, NRM2, ASUM, AMAX) |
| xgemv | Level-2 matrix-vector ops |
| xger | Level-2 rank-1 update ops |
| xgemm | Level-3 matrix-matrix ops (GEMM, SYMM, HERM, SYRK, HERK) |
| xgemm_direct | Direct-unrolled GEMM variant |
| copy | Matrix copy operations |
| transpose | Matrix transpose operations |
| invert | Triangular solve (TRSM) |
| pad / padtranspose | Padding operations for convolution |
| xconvgemm | Convolution-as-GEMM operations |

### Routine Tuners (`src/tuning/routines/`)

- `xgemm.cpp` – End-to-end GEMM tuning across multiple kernel configurations
- `xtrsv.cpp` – Triangular solve tuning

---

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

- **`build_and_test.yml`**: Builds on Ubuntu (gcc, clang) and macOS (clang)
- **`lint.yml`**: Code style checks
- **`release.yml`**: Release artifact creation

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `include/clblast_vk.h` | Main public API header – all BLAS routine declarations |
| `src/clblast_vk.cpp` | Vulkan implementation of the BLAS API |
| `src/vkpp11.hpp` | OpenCL-like C++11 wrapper over Tart |
| `src/routine.hpp` | Base class for all BLAS routine implementations |
| `src/routine_aot_cache.hpp` | Precompiled kernel accessor functions |
| `src/cache.hpp` | Kernel compile-time caching |
| `src/database/database.hpp` | Database initialization and lookup |
| `src/database/database_structure.hpp` | Database data structures |
| `src/utilities/backend.hpp` | Backend selector (always Vulkan) |
| `src/tuning/tuning_api.cpp` | Public tuning API |
| `src/tuning/tuning.cpp` | Tuning engine core |
| `src/tuning/configurations.hpp` | Kernel parameter configurations |
| `src/kernels-vk/` | GLSL compute shader sources (organized by BLAS level) |
| `tart/include/tart/tart.hpp` | Tart public API header |
| `tart/src/device.cpp` | Tart device implementation |
| `tart/src/shader.cpp` | Tart shader/pipeline/kernel implementation |
| `tart/src/gpu.cpp` | Tart buffer/event implementation |
| `tart/src/compilers.cpp` | Tart shader compilation (GLSL/clspv) |
| `tart/CMakeLists.txt` | Tart build configuration |
| `CMakeLists.txt` | Root CLBlast-vk build configuration |
| `cmake/compile_kernels.cmake` | Kernel source generation macro |
| `scripts/inline_gen.py` | SPIR-V inline header generation |
| `vcpkg.json` | vcpkg manifest (no longer has external deps) |

---

## Coding Conventions

### CLBlast-vk (`src/`)

- **Language**: C++17
- **Namespace**: `clblast`
- **Naming**: PascalCase for classes, camelCase for methods/variables, `m` prefix for members
- **Error handling**: Throws `std::runtime_error` with descriptive messages
- **Templates**: All BLAS routines are templated on precision type (`T`)
- **Backend guards**: `#if VULKAN_API` / `#elif OPENCL_API` sections

### Tart (`tart/`)

- **Language**: C++17
- **Namespace**: `tart`
- **Naming**: camelCase for methods, PascalCase for classes, `m` prefix for members
- **Pointers**: `device_ptr` (shared_ptr), `device_ref` (weak_ptr) throughout
- **PIMPL**: Internal classes use `std::unique_ptr` owned by Device, forward-declared in public header
- **Thread safety**: `std::recursive_mutex` in Device class
- **Shader language**: GLSL #version 450 compute shaders

---

## TODO Items (from tart/README.md)

- [ ] Refactor device creation code (adapted from GGML)
- [ ] Implement remaining Vulkan extensions for shader-side functionality
- [ ] Improve cross-platform Python bindings (scikit-build-core in progress)
- [ ] Simplify compile-time definitions in shader sources
- [ ] Clean up `tart::CLProgram` and `tart::CLKernel` (legacy OpenCL)
- [ ] Implement host-side support for uncommon quantized data types (float16, float8, bfloat16)
- [ ] Benchmark infrastructure (`TART_ENABLE_BENCHMARKS` is a placeholder)
- [ ] Full BLAS library under Tart (`TART_BUILD_BLAS` not implemented yet)

---

## Quick Start Examples

### Minimal CLBlast-vk Usage (C++)

```cpp
#include <clblast_vk.h>
#include <tart.hpp>

// Initialize Vulkan device
tart::Instance instance;
auto device = instance.getDevice(0);

// Allocate buffers
auto a_buf = device->allocateBuffer(m * k * sizeof(float));
auto b_buf = device->allocateBuffer(k * n * sizeof(float));
auto c_buf = device->allocateBuffer(m * n * sizeof(float));

// Copy data to device
device->copyIn(a_buf, host_a.data(), m * k * sizeof(float));
device->copyIn(b_buf, host_b.data(), k * n * sizeof(float));

// Run GEMM: C = alpha * A * B + beta * C
clblast::Gemm<float>(
    clblast::Layout::kRowMajor,
    clblast::Transpose::kNo, clblast::Transpose::kNo,
    m, n, k,
    1.0f, a_buf, 0, k,  // A matrix
    b_buf, 0, n,         // B matrix
    0.0f, c_buf, 0, n,   // C matrix
    device
);

// Read result
device->copyOut(result, c_buf, m * n * sizeof(float));
```

### Minimal Tart Usage (C++)

```cpp
#include <tart/tart.hpp>

// Initialize
tart::Instance instance;
auto device = instance.getDevice(0);

// Allocate buffer
auto buf = device->allocateBuffer(size_in_bytes);

// Compile GLSL shader
auto shader = device->compileGLSL("#version 450\nlayout(local_size_x = 256) in\nvoid main() {\n  // work\n}\n");

// Create and dispatch pipeline
auto pipeline = device->createPipeline(shader, "main");
device->enqueuePipeline(pipeline, {num_threads}, {buf});
device->sync();
```

### Python (pytart)

```python
import pytart

instance = pytart.Instance()
device = instance.get_device(0)

buf = device.allocate_buffer(size)
buf.copy_in(host_data)

shader = device.compile_glsl("#version 450\n...")
pipeline = device.create_pipeline(shader, "main")
device.dispatch_pipeline(pipeline, [num_threads], [buf])
device.sync()

result = buf.copy_out()
```
