
# ==================================================================================================
# compile_kernels.cmake – Generate AOT kernel sources using tart-spirv-inliner
# ==================================================================================================
#
# The new system uses Tart's tart_add_kernel_set() macro which calls the
# tart-spirv-inliner tool. Each invocation produces one .hpp/.cpp bundle
# containing kernels for all 5 namespaces (f16, f32, f64, c64, c128).
#
# The tool derives the kernel name from the input file's stem, so we create
# symlinks with the desired kernel names pointing to the shared GLSL sources.
# This allows us to batch all 5 namespaces for one kernel group into a
# single invocation.
#
# Old system: ~500 calls to tart_build_kernel_spv (one per namespace/kernel)
# New system: ~40 calls to tart_add_kernel_set (one per unique source+args)
# ==================================================================================================

# -----------------------------------------------------------------------
# Helper: generate symlinks for variants that share the same source file
# but need different kernel names (e.g., axpy vs axpy_faster vs axpy_fastest)
# or different compiler args (e.g., different vector widths).
# -----------------------------------------------------------------------
function(tart_generate_variant_symlinks SYMLINK_DIR SOURCE_FILE BASE_NAME)
    set(VW 1 2 4 8 16)
    set(VECTOR_WIDTHS ${VW})
    set(BINARY_OPTIONS 0 1)

    foreach(VW_VAL ${VECTOR_WIDTHS})
        file(CREATE_LINK
            "${SOURCE_FILE}"
            "${SYMLINK_DIR}/${BASE_NAME}_vw${VW_VAL}.glsl"
            SYMBOLIC
        )
    endforeach()

    # Faster and fastest variants share the same source
    foreach(VAR_SUFFIX faster fastest)
        foreach(VW_VAL ${VECTOR_WIDTHS})
            file(CREATE_LINK
                "${SOURCE_FILE}"
                "${SYMLINK_DIR}/${BASE_NAME}_${VAR_SUFFIX}_vw${VW_VAL}.glsl"
                SYMBOLIC
            )
        endforeach()
    endforeach()

    # Subgroup variants (0 and 1) share the same source
    foreach(SUB ${BINARY_OPTIONS})
        file(CREATE_LINK
            "${SOURCE_FILE}"
            "${SYMLINK_DIR}/${BASE_NAME}_sub${SUB}.glsl"
            SYMBOLIC
        )
    endforeach()

    # Gemm direct dual-vector-width variants (1..16 x 1..16 = 25 files)
    foreach(VWM ${VECTOR_WIDTHS})
        foreach(VWN ${VECTOR_WIDTHS})
            file(CREATE_LINK
                "${SOURCE_FILE}"
                "${SYMLINK_DIR}/${BASE_NAME}_vw${VWM}_vw${VWN}.glsl"
                SYMBOLIC
            )
        endforeach()
    endforeach()
endfunction()

# -----------------------------------------------------------------------
# Helper: invoke tart_add_kernel_set for a kernel group
# First positional arg is the keyword NAME, second is the kernel set name
# -----------------------------------------------------------------------
function(tart_compile_kernel_group)
    # ARGV0=NAME keyword, ARGV1=actual kernel set name
    set(KERNEL_SET_NAME ${ARGV1})
    cmake_parse_arguments(GROUP "" "" "INPUTS;COMPILER_ARGS" ${ARGN})

    message(STATUS "Kernel group: ${KERNEL_SET_NAME}")

    tart_add_kernel_set(
        NAME ${KERNEL_SET_NAME}
        INPUTS ${GROUP_INPUTS}
        COMPILER_ARGS ${GROUP_COMPILER_ARGS}
    )
endfunction()

# -----------------------------------------------------------------------
# Namespace list
# -----------------------------------------------------------------------
set(ALL_KERNEL_NAMESPACES f16 f32 f64 c64 c128)
set(VECTOR_WIDTHS 1 2 4 8 16)
set(BINARY_FLAG_OPTIONS 0 1)

# Create symlink directory for variant files
set(TART_KERNEL_SYMLINK_DIR "${PROJECT_BINARY_DIR}/kernels/inline")
file(MAKE_DIRECTORY "${TART_KERNEL_SYMLINK_DIR}")

# Generate symlinks for variant groups (these are shared across the groups below)
# Level 1 AXPY/COPY/SCAL/SWAP/HAD variants
# Note: GLSL files use x-prefix (xaxpy.glsl) but kernels use no-prefix (axpy)
# Copy shared GLSL includes to the inline directory so relative includes resolve
file(COPY
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/common.glsl"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/dtype-enum.glsl"
    DESTINATION "${TART_KERNEL_SYMLINK_DIR}/"
)
# Copy aggregator files that other files include
file(COPY
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/level1.glsl"
    DESTINATION "${TART_KERNEL_SYMLINK_DIR}/"
)
file(COPY
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/level2.glsl"
    DESTINATION "${TART_KERNEL_SYMLINK_DIR}/"
)
file(COPY
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/level3.glsl"
    DESTINATION "${TART_KERNEL_SYMLINK_DIR}/"
)
file(COPY
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/level3_fill_matrix.glsl"
    DESTINATION "${TART_KERNEL_SYMLINK_DIR}/"
)
# Copy all source files from level1 and levelx (includes sub-includes like level1.glsl)
file(GLOB LEVEL1_GLSL "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/*.glsl")
file(COPY ${LEVEL1_GLSL} DESTINATION "${TART_KERNEL_SYMLINK_DIR}/")
file(GLOB LEVELX_GLSL "${PROJECT_SOURCE_DIR}/src/kernels-vk/levelx/*.glsl")
file(COPY ${LEVELX_GLSL} DESTINATION "${TART_KERNEL_SYMLINK_DIR}/")
# Copy level3 files (includes sub-includes like xgemm_part*.glsl)
file(GLOB LEVEL3_GLSL "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/*.glsl")
file(COPY ${LEVEL3_GLSL} DESTINATION "${TART_KERNEL_SYMLINK_DIR}/")
# Copy level3 files to kernels/ so ../level3/ includes from levelx resolve
file(COPY
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/"
    DESTINATION "${CMAKE_BINARY_DIR}/kernels/level3/"
)
# Copy level2 files (includes sub-includes like level2.glsl)
file(GLOB LEVEL2_GLSL "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/*.glsl")
file(COPY ${LEVEL2_GLSL} DESTINATION "${TART_KERNEL_SYMLINK_DIR}/")
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl"
    "axpy"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xcopy.glsl"
    "copy"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xcopy_fast.glsl"
    "copy_fast"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xscal.glsl"
    "scal"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xscal_fast.glsl"
    "scal_fast"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xswap.glsl"
    "swap"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xswap_fast.glsl"
    "swap_fast"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xhad.glsl"
    "had"
)

# Reduction kernels: code expects names without x-prefix (amax, asum, dot, nrm2)
# Create symlinks without x-prefix that point to x-prefixed source files
foreach(KERNEL amax asum dot nrm2)
    file(CREATE_LINK
        "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl"
        "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
        SYMBOLIC
    )
    file(CREATE_LINK
        "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_epilogue.glsl"
        "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_epilogue.glsl"
        SYMBOLIC
    )
endforeach()

# Level 2 GEMV fast variants
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xgemv.glsl"
    "gemv"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xgemv_fast.glsl"
    "gemv_fast"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xgemv_fast_rot.glsl"
    "gemv_fast_rot"
)

# Level 3 GEMM variants
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/xgemm.glsl"
    "gemm"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/xgemm_direct.glsl"
    "gemm_direct"
)

# Dual vector-width symlinks for GEMM direct variants (25 combinations per kernel)
set(DUAL_VW_KERNELS gemm_direct gemm_direct_batched gemm_direct_strided_batched)
set(DUAL_VW_SOURCES
    gemm_direct xgemm_direct.glsl
    gemm_direct_batched xgemm_direct_batched.glsl
    gemm_direct_strided_batched xgemm_direct_strided_batched.glsl
)
list(LENGTH DUAL_VW_SOURCES _DUAL_LIST_LEN)
math(EXPR _DUAL_LAST_IDX "${_DUAL_LIST_LEN} - 1")
foreach(_DUAL_IDX RANGE 0 ${_DUAL_LAST_IDX} 2)
    list(GET DUAL_VW_SOURCES ${_DUAL_IDX} _DKERNEL)
    math(EXPR _DFILE_IDX "${_DUAL_IDX} + 1")
    list(GET DUAL_VW_SOURCES ${_DFILE_IDX} _DSOURCE)
    foreach(VWM IN LISTS VECTOR_WIDTHS)
        foreach(VWN IN LISTS VECTOR_WIDTHS)
            file(CREATE_LINK
                "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/${_DSOURCE}"
                "${TART_KERNEL_SYMLINK_DIR}/${_DKERNEL}_vw${VWM}_vw${VWN}.glsl"
                SYMBOLIC
            )
        endforeach()
    endforeach()
endforeach()

# Level 3 TRANSPOSE fast variants
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/transpose_fast.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/transpose_fast_vw1.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/transpose_fast.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/transpose_fast_vw2.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/transpose_fast.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/transpose_fast_vw4.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/transpose_fast.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/transpose_fast_vw8.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/transpose_fast.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/transpose_fast_vw16.glsl"
    SYMBOLIC
)

# LevelX IM2COL/COL2IM variants
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/levelx/im2col_kernel_normal.glsl"
    "im2col_kernel_normal"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/levelx/col2im_kernel_normal.glsl"
    "col2im_kernel_normal"
)

# LevelX CONVGEMM variants (two different source files)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/levelx/convgemm_kernel.glsl"
    "convgemm_kernel"
)
tart_generate_variant_symlinks(
    "${TART_KERNEL_SYMLINK_DIR}"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/levelx/convgemm_normal.glsl"
    "convgemm_normal"
)

# ==================================================================================================
# Kernel groups – each group shares the same compiler arguments
# ==================================================================================================

# Global include paths for GLSL shaders
set(TART_GLSL_INCLUDE_PATHS "-I${PROJECT_SOURCE_DIR}/src/kernels-vk" "-I${CMAKE_BINARY_DIR}/kernels")

# --- Level 1: Vector-width variants (axpy, copy, scal, swap, had) ---
# These use -DVW=<width> as a per-input compiler arg
set(L1_VW_KERNELS axpy copy scal swap had)
foreach(KERNEL ${L1_VW_KERNELS})
    set(_INPUTS "")
    set(_COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS})
    foreach(VW ${VECTOR_WIDTHS})
        list(APPEND _INPUTS f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw${VW}.glsl"
                           f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw${VW}.glsl"
                           f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw${VW}.glsl"
                           c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw${VW}.glsl"
                           c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw${VW}.glsl")
        list(APPEND _COMPILER_ARGS "-DVW=${VW}")
    endforeach()
    tart_compile_kernel_group(
        NAME "all-blas-kernels-${KERNEL}_vw"
        INPUTS ${_INPUTS}
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128 ${_COMPILER_ARGS}
    )
endforeach()

# --- Level 1: Faster and fastest variants (axpy, copy, scal, swap, had) ---
foreach(VAR_SUFFIX faster fastest)
    set(L1_FAST_KERNELS axpy copy scal swap had)
    foreach(KERNEL ${L1_FAST_KERNELS})
        set(_INPUTS "")
        set(_COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS})
        foreach(VW ${VECTOR_WIDTHS})
            list(APPEND _INPUTS f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_${VAR_SUFFIX}_vw${VW}.glsl"
                               f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_${VAR_SUFFIX}_vw${VW}.glsl"
                               f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_${VAR_SUFFIX}_vw${VW}.glsl"
                               c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_${VAR_SUFFIX}_vw${VW}.glsl"
                               c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_${VAR_SUFFIX}_vw${VW}.glsl")
            list(APPEND _COMPILER_ARGS "-DVW=${VW}")
        endforeach()
        tart_compile_kernel_group(
            NAME "all-blas-kernels-${KERNEL}_${VAR_SUFFIX}"
            INPUTS ${_INPUTS}
            COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128 ${_COMPILER_ARGS}
        )
    endforeach()
endforeach()

# --- Level 1: Fast variants (copy_fast, scal_fast, swap_fast) ---
set(L1_FAST_KERNELS copy scal swap)
foreach(KERNEL ${L1_FAST_KERNELS})
    set(_INPUTS "")
    set(_COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS})
    foreach(VW ${VECTOR_WIDTHS})
        list(APPEND _INPUTS f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_fast_vw${VW}.glsl"
                           f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_fast_vw${VW}.glsl"
                           f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_fast_vw${VW}.glsl"
                           c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_fast_vw${VW}.glsl"
                           c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_fast_vw${VW}.glsl")
        list(APPEND _COMPILER_ARGS "-DVW=${VW}")
    endforeach()
    tart_compile_kernel_group(
        NAME "all-blas-kernels-${KERNEL}_fast"
        INPUTS ${_INPUTS}
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128 ${_COMPILER_ARGS}
    )
endforeach()

# --- Level 1: Batched (axpy_batched) – no vector width ---
tart_compile_kernel_group(
    NAME "all-blas-kernels-axpy_batched"
    INPUTS
        f16 "${TART_KERNEL_SYMLINK_DIR}/axpy_batched.glsl"
        f32 "${TART_KERNEL_SYMLINK_DIR}/axpy_batched.glsl"
        f64 "${TART_KERNEL_SYMLINK_DIR}/axpy_batched.glsl"
        c64 "${TART_KERNEL_SYMLINK_DIR}/axpy_batched.glsl"
        c128 "${TART_KERNEL_SYMLINK_DIR}/axpy_batched.glsl"
    COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128
)

# --- Level 2: Fixed kernels (gemv, ger, her, her2, trsv_forward, trsv_backward, trsv_fill_vector) ---
set(KERNELS_LEVEL2 gemv ger her her2 trsv_forward trsv_backward trsv_fill_vector)
foreach(KERNEL ${KERNELS_LEVEL2})
    tart_compile_kernel_group(
        NAME "all-blas-kernels-${KERNEL}"
        INPUTS
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128
    )
endforeach()

# --- Level 2: Fast GEMV variants (gemv_fast, gemv_fast_rot) ---
foreach(KERNEL gemv_fast gemv_fast_rot)
    set(_INPUTS "")
    set(_COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS})
    foreach(VW ${VECTOR_WIDTHS})
        list(APPEND _INPUTS f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw${VW}.glsl"
                           f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw${VW}.glsl"
                           f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw${VW}.glsl"
                           c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw${VW}.glsl"
                           c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw${VW}.glsl")
        list(APPEND _COMPILER_ARGS "-DVW=${VW}")
    endforeach()
    tart_compile_kernel_group(
        NAME "all-blas-kernels-${KERNEL}"
        INPUTS ${_INPUTS}
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128 ${_COMPILER_ARGS}
    )
endforeach()

# --- Level 1: Reduction kernels (amax, asum, dot, nrm2) ---
set(REDUCE_KERNELS amax asum dot nrm2)
foreach(KERNEL ${REDUCE_KERNELS})
    tart_compile_kernel_group(
        NAME "all-blas-kernels-${KERNEL}"
        INPUTS
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128
    )
endforeach()

# --- Level 1: Epilogue kernels (amax_epilogue, asum_epilogue, dot_epilogue, nrm2_epilogue) ---
foreach(KERNEL ${REDUCE_KERNELS})
    tart_compile_kernel_group(
        NAME "all-blas-kernels-${KERNEL}_epilogue"
        INPUTS
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_epilogue.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_epilogue.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_epilogue.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_epilogue.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_epilogue.glsl"
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128
    )
endforeach()

# --- Level 1: Batched and Level 2: Symlinks without x-prefix ---
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_batched.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/axpy_batched.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xgemv.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/gemv.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xgemv_fast.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/gemv_fast.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xgemv_fast_rot.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/gemv_fast_rot.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xger.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/ger.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xher.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/her.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xher2.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/her2.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xtrsv_forward.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/trsv_forward.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xtrsv_backward.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/trsv_backward.glsl"
    SYMBOLIC
)
file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/xtrsv_fill_vector.glsl"
    "${TART_KERNEL_SYMLINK_DIR}/trsv_fill_vector.glsl"
    SYMBOLIC
)

# --- Level 3: No-arg/No-X kernels (convert_*, fill_matrix, transpose_*, copy_*) ---
set(LEVEL3_NOARG_KERNELS
    convert_hermitian_upper convert_hermitian_lower
    convert_symmetric_upper convert_symmetric_lower
    convert_triangular_upper convert_triangular_lower
    level3_fill_matrix
    transpose_pad_matrix transpose_pad_matrix_batched transpose_pad_matrix_strided_batched
    transpose_matrix transpose_matrix_batched transpose_matrix_strided_batched
    copy_matrix_fast copy_pad_matrix copy_pad_matrix_batched copy_pad_matrix_strided_batched
    copy_matrix copy_matrix_batched copy_matrix_strided_batched
)
foreach(KERNEL ${LEVEL3_NOARG_KERNELS})
    tart_compile_kernel_group(
        NAME "all-blas-kernels-${KERNEL}"
        INPUTS
            f16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/${KERNEL}.glsl"
            f32 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/${KERNEL}.glsl"
            f64 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/${KERNEL}.glsl"
            c64 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/${KERNEL}.glsl"
            c128 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/${KERNEL}.glsl"
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128
    )
endforeach()

# --- Level 3: GEMM subgroup variants (gemm, gemm_upper, gemm_lower, gemm_batched, gemm_strided_batched) ---
# Each has sub0 (no subgroup ops) and sub1 (with subgroup ops) variants
# Need to generate symlinks for each variant from their specific source files
set(LEVEL3_SUBGROUP_SOURCE_FILES
    gemm xgemm.glsl
    gemm_upper xgemm_upper.glsl
    gemm_lower xgemm_lower.glsl
    gemm_batched xgemm_batched.glsl
    gemm_strided_batched xgemm_strided_batched.glsl
)
list(LENGTH LEVEL3_SUBGROUP_SOURCE_FILES _LIST_LEN)
math(EXPR _LAST_IDX "${_LIST_LEN} - 1")
foreach(_IDX RANGE 0 ${_LAST_IDX} 2)
    list(GET LEVEL3_SUBGROUP_SOURCE_FILES ${_IDX} KERNEL_NAME)
    math(EXPR _FILE_IDX "${_IDX} + 1")
    list(GET LEVEL3_SUBGROUP_SOURCE_FILES ${_FILE_IDX} SOURCE_FILE)
    foreach(SUB ${BINARY_FLAG_OPTIONS})
        file(CREATE_LINK
            "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/${SOURCE_FILE}"
            "${TART_KERNEL_SYMLINK_DIR}/${KERNEL_NAME}_sub${SUB}.glsl"
            SYMBOLIC
        )
    endforeach()
endforeach()

# --- Dual vector-width symlinks for subgroup variants ---
# Creates e.g. gemm_sub0_vw1_vw1.glsl -> gemm_sub0.glsl for all VW combos
set(LEVEL3_SUBGROUP_KERNELS gemm gemm_upper gemm_lower gemm_batched gemm_strided_batched)
foreach(_SK ${LEVEL3_SUBGROUP_KERNELS})
    foreach(_SUB ${BINARY_FLAG_OPTIONS})
        list(APPEND _DUAL_SUB_GROUP "${_SK}_sub${_SUB}.glsl")
    endforeach()
endforeach()
list(LENGTH _DUAL_SUB_GROUP _DUAL_SUB_LEN)
math(EXPR _DUAL_SUB_LAST "${_DUAL_SUB_LEN} - 1")
foreach(_DSI RANGE 0 ${_DUAL_SUB_LAST} 1)
    list(GET _DUAL_SUB_GROUP ${_DSI} _DSRC)
    string(REGEX REPLACE "\.glsl$" "" _DSK "${_DSRC}")
    foreach(_VWM ${VECTOR_WIDTHS})
        foreach(_VWN ${VECTOR_WIDTHS})
            file(CREATE_LINK
                "${TART_KERNEL_SYMLINK_DIR}/${_DSRC}"
                "${TART_KERNEL_SYMLINK_DIR}/${_DSK}_vw${_VWM}_vw${_VWN}.glsl"
                SYMBOLIC
            )
        endforeach()
    endforeach()
endforeach()

# --- Dual vector-width compilation for subgroup variants ---
foreach(_SK ${LEVEL3_SUBGROUP_KERNELS})
    foreach(_SUB ${BINARY_FLAG_OPTIONS})
        tart_compile_kernel_group(
            NAME "all-blas-kernels-${_SK}_sub${_SUB}"
            INPUTS
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw1.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw1.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw1.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw1.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw1.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw2.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw2.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw2.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw2.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw2.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw4.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw4.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw4.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw4.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw4.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw8.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw8.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw8.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw8.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw8.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw16.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw16.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw16.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw16.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw1_vw16.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw1.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw1.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw1.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw1.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw1.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw2.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw2.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw2.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw2.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw2.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw4.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw4.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw4.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw4.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw4.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw8.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw8.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw8.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw8.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw8.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw16.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw16.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw16.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw16.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw2_vw16.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw1.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw1.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw1.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw1.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw1.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw2.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw2.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw2.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw2.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw2.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw4.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw4.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw4.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw4.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw4.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw8.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw8.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw8.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw8.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw8.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw16.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw16.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw16.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw16.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw4_vw16.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw1.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw1.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw1.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw1.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw1.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw2.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw2.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw2.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw2.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw2.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw4.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw4.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw4.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw4.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw4.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw8.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw8.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw8.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw8.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw8.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw16.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw16.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw16.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw16.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw8_vw16.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw1.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw1.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw1.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw1.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw1.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw2.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw2.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw2.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw2.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw2.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw4.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw4.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw4.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw4.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw4.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw8.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw8.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw8.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw8.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw8.glsl"
                f16 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw16.glsl"
                f32 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw16.glsl"
                f64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw16.glsl"
                c64 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw16.glsl"
                c128 "${TART_KERNEL_SYMLINK_DIR}/${_SK}_sub${_SUB}_vw16_vw16.glsl"
            COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128 -DVWM=1 -DVWN=1 -DSUBGROUP_OPERATIONS_SUPPORTED=${_SUB} --target-env=vulkan1.2
        )
    endforeach()
endforeach()

# --- Level 3: GEMM direct dual-vector-width variants ---
set(LEVEL3_DUAL_VW_KERNELS gemm_direct gemm_direct_batched gemm_direct_strided_batched)
foreach(KERNEL ${LEVEL3_DUAL_VW_KERNELS})
    tart_compile_kernel_group(
        NAME "all-blas-kernels-${KERNEL}"
        INPUTS
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw1.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw1.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw1.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw1.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw1.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw2.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw2.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw2.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw2.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw2.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw4.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw4.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw4.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw4.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw4.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw8.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw8.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw8.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw8.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw8.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw16.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw16.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw16.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw16.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw16.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw1.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw1.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw1.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw1.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw1.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw2.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw2.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw2.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw2.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw2.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw4.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw4.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw4.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw4.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw4.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw8.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw8.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw8.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw8.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw8.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw16.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw16.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw16.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw16.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw16.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw1.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw1.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw1.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw1.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw1.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw2.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw2.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw2.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw2.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw2.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw4.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw4.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw4.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw4.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw4.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw8.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw8.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw8.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw8.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw8.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw16.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw16.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw16.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw16.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw16.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw1.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw1.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw1.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw1.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw1.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw2.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw2.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw2.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw2.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw2.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw4.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw4.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw4.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw4.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw4.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw8.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw8.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw8.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw8.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw8.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw16.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw16.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw16.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw16.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw16.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw1.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw1.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw1.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw1.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw1.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw2.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw2.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw2.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw2.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw2.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw4.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw4.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw4.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw4.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw4.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw8.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw8.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw8.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw8.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw8.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw16.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw16.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw16.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw16.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw16.glsl"
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128 -DVWM=1 -DVWN=1
    )
endforeach()

# --- Level 3: Transpose fast variants (-DTRA_WPT=<vw>) ---
foreach(VW ${VECTOR_WIDTHS})
    tart_compile_kernel_group(
        NAME "all-blas-kernels-transpose_fast_vw${VW}"
        INPUTS
            f16 "${TART_KERNEL_SYMLINK_DIR}/transpose_fast_vw${VW}.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/transpose_fast_vw${VW}.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/transpose_fast_vw${VW}.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/transpose_fast_vw${VW}.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/transpose_fast_vw${VW}.glsl"
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128 -DTRA_WPT=${VW}
    )
endforeach()

# --- LevelX: IM2COL/COL2IM (no vector width) ---
set(LEVELX_KERNELS im2col_kernel_normal col2im_kernel_normal)
foreach(KERNEL ${LEVELX_KERNELS})
    tart_compile_kernel_group(
        NAME "all-blas-kernels-${KERNEL}"
        INPUTS
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}.glsl"
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128
    )
endforeach()

# --- LevelX: CONVGEMM (dual vector width) ---
set(LEVELX_CONVGEMM_KERNELS convgemm_kernel convgemm_normal)
foreach(KERNEL ${LEVELX_CONVGEMM_KERNELS})
    tart_compile_kernel_group(
        NAME "all-blas-kernels-${KERNEL}"
        INPUTS
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw1.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw1.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw1.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw1.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw1.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw2.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw2.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw2.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw2.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw2.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw4.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw4.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw4.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw4.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw4.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw8.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw8.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw8.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw8.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw8.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw16.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw16.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw16.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw16.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw1_vw16.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw1.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw1.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw1.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw1.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw1.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw2.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw2.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw2.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw2.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw2.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw4.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw4.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw4.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw4.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw4.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw8.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw8.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw8.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw8.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw8.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw16.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw16.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw16.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw16.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw2_vw16.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw1.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw1.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw1.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw1.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw1.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw2.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw2.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw2.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw2.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw2.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw4.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw4.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw4.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw4.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw4.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw8.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw8.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw8.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw8.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw8.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw16.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw16.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw16.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw16.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw4_vw16.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw1.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw1.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw1.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw1.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw1.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw2.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw2.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw2.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw2.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw2.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw4.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw4.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw4.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw4.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw4.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw8.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw8.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw8.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw8.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw8.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw16.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw16.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw16.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw16.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw8_vw16.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw1.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw1.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw1.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw1.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw1.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw2.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw2.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw2.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw2.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw2.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw4.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw4.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw4.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw4.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw4.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw8.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw8.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw8.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw8.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw8.glsl"
            f16 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw16.glsl"
            f32 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw16.glsl"
            f64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw16.glsl"
            c64 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw16.glsl"
            c128 "${TART_KERNEL_SYMLINK_DIR}/${KERNEL}_vw16_vw16.glsl"
        COMPILER_ARGS ${TART_GLSL_INCLUDE_PATHS} -DDTYPE=DTYPE_F16 -DDTYPE=DTYPE_F32 -DDTYPE=DTYPE_F64 -DDTYPE=DTYPE_C64 -DDTYPE=DTYPE_C128 -DVWM=1 -DVWN=1
    )
endforeach()

# ==================================================================================================
# Kernel set tracking – lists all tart_add_kernel_set() invocations for CMakeVariables.cmake to wire
# into the clblast library build. Each set produces one .hpp + .cpp bundle.
# ==================================================================================================
set(TART_AOT_KERNEL_SETS
    all-blas-kernels-axpy_vw
    all-blas-kernels-copy_vw
    all-blas-kernels-scal_vw
    all-blas-kernels-swap_vw
    all-blas-kernels-had_vw
    all-blas-kernels-axpy_faster
    all-blas-kernels-copy_fast
    all-blas-kernels-scal_fast
    all-blas-kernels-swap_fast
    all-blas-kernels-had_faster
    all-blas-kernels-axpy_fastest
    all-blas-kernels-copy_fastest
    all-blas-kernels-scal_fastest
    all-blas-kernels-swap_fastest
    all-blas-kernels-had_fastest
    all-blas-kernels-axpy_batched
    all-blas-kernels-gemv
    all-blas-kernels-gemv_fast
    all-blas-kernels-gemv_fast_rot
    all-blas-kernels-ger
    all-blas-kernels-her
    all-blas-kernels-her2
    all-blas-kernels-trsv_forward
    all-blas-kernels-trsv_backward
    all-blas-kernels-trsv_fill_vector
    all-blas-kernels-amax
    all-blas-kernels-asum
    all-blas-kernels-dot
    all-blas-kernels-nrm2
    all-blas-kernels-amax_epilogue
    all-blas-kernels-asum_epilogue
    all-blas-kernels-dot_epilogue
    all-blas-kernels-nrm2_epilogue
    all-blas-kernels-convert_hermitian_upper
    all-blas-kernels-convert_hermitian_lower
    all-blas-kernels-convert_symmetric_upper
    all-blas-kernels-convert_symmetric_lower
    all-blas-kernels-convert_triangular_upper
    all-blas-kernels-convert_triangular_lower
    all-blas-kernels-level3_fill_matrix
    all-blas-kernels-transpose_pad_matrix
    all-blas-kernels-transpose_pad_matrix_batched
    all-blas-kernels-transpose_pad_matrix_strided_batched
    all-blas-kernels-transpose_matrix
    all-blas-kernels-transpose_matrix_batched
    all-blas-kernels-transpose_matrix_strided_batched
    all-blas-kernels-copy_matrix_fast
    all-blas-kernels-copy_pad_matrix
    all-blas-kernels-copy_pad_matrix_batched
    all-blas-kernels-copy_pad_matrix_strided_batched
    all-blas-kernels-copy_matrix
    all-blas-kernels-copy_matrix_batched
    all-blas-kernels-copy_matrix_strided_batched
    all-blas-kernels-gemm_sub0
    all-blas-kernels-gemm_sub1
    all-blas-kernels-gemm_upper_sub0
    all-blas-kernels-gemm_upper_sub1
    all-blas-kernels-gemm_lower_sub0
    all-blas-kernels-gemm_lower_sub1
    all-blas-kernels-gemm_batched_sub0
    all-blas-kernels-gemm_batched_sub1
    all-blas-kernels-gemm_strided_batched_sub0
    all-blas-kernels-gemm_strided_batched_sub1
    all-blas-kernels-gemm_direct
    all-blas-kernels-gemm_direct_batched
    all-blas-kernels-gemm_direct_strided_batched
    all-blas-kernels-transpose_fast_vw1
    all-blas-kernels-transpose_fast_vw2
    all-blas-kernels-transpose_fast_vw4
    all-blas-kernels-transpose_fast_vw8
    all-blas-kernels-transpose_fast_vw16
    all-blas-kernels-im2col_kernel_normal
    all-blas-kernels-col2im_kernel_normal
    all-blas-kernels-convgemm_kernel
    all-blas-kernels-convgemm_normal
)
