

# most args will be consistent, so wrap it
macro(build_kernel KERNEL_NAMESPACE KERNEL_NAME INPUT_FILENAME BUILD_ARGS)
	message(${BUILD_ARGS})
	tart_build_kernel_spv(${KERNEL_NAMESPACE} ${KERNEL_NAME} main glsl ${INPUT_FILENAME} "${BUILD_ARGS}")
endmacro()

# axpy
build_kernel(f16 axpy_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F16 -DVW=1")
build_kernel(f16 axpy_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F16 -DVW=2")
build_kernel(f16 axpy_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F16 -DVW=4")
build_kernel(f16 axpy_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F16 -DVW=8")
build_kernel(f16 axpy_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F16 -DVW=16")

build_kernel(f32 axpy_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F32 -DVW=1")
build_kernel(f32 axpy_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F32 -DVW=2")
build_kernel(f32 axpy_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F32 -DVW=4")
build_kernel(f32 axpy_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F32 -DVW=8")
build_kernel(f32 axpy_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F32 -DVW=16")

build_kernel(f64 axpy_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F64 -DVW=1")
build_kernel(f64 axpy_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F64 -DVW=2")
build_kernel(f64 axpy_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F64 -DVW=4")
build_kernel(f64 axpy_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F64 -DVW=8")
build_kernel(f64 axpy_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F64 -DVW=16")

build_kernel(c64 axpy_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C64 -DVW=1")
build_kernel(c64 axpy_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C64 -DVW=2")
build_kernel(c64 axpy_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C64 -DVW=4")
build_kernel(c64 axpy_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C64 -DVW=8")
build_kernel(c64 axpy_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C64 -DVW=16")

build_kernel(c128 axpy_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C128 -DVW=1")
build_kernel(c128 axpy_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C128 -DVW=2")
build_kernel(c128 axpy_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C128 -DVW=4")
build_kernel(c128 axpy_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C128 -DVW=8")
build_kernel(c128 axpy_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C128 -DVW=16")

# axpy faster
build_kernel(f16 axpy_faster_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F16 -DVW=1")
build_kernel(f16 axpy_faster_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F16 -DVW=2")
build_kernel(f16 axpy_faster_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F16 -DVW=4")
build_kernel(f16 axpy_faster_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F16 -DVW=8")
build_kernel(f16 axpy_faster_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F16 -DVW=16")

build_kernel(f32 axpy_faster_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F32 -DVW=1")
build_kernel(f32 axpy_faster_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F32 -DVW=2")
build_kernel(f32 axpy_faster_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F32 -DVW=4")
build_kernel(f32 axpy_faster_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F32 -DVW=8")
build_kernel(f32 axpy_faster_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F32 -DVW=16")

build_kernel(f64 axpy_faster_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F64 -DVW=1")
build_kernel(f64 axpy_faster_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F64 -DVW=2")
build_kernel(f64 axpy_faster_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F64 -DVW=4")
build_kernel(f64 axpy_faster_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F64 -DVW=8")
build_kernel(f64 axpy_faster_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F64 -DVW=16")

build_kernel(c64 axpy_faster_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C64 -DVW=1")
build_kernel(c64 axpy_faster_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C64 -DVW=2")
build_kernel(c64 axpy_faster_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C64 -DVW=4")
build_kernel(c64 axpy_faster_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C64 -DVW=8")
build_kernel(c64 axpy_faster_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C64 -DVW=16")

build_kernel(c128 axpy_faster_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C128 -DVW=1")
build_kernel(c128 axpy_faster_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C128 -DVW=2")
build_kernel(c128 axpy_faster_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C128 -DVW=4")
build_kernel(c128 axpy_faster_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C128 -DVW=8")
build_kernel(c128 axpy_faster_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C128 -DVW=16")

# axpy fastest
build_kernel(f16 axpy_fastest_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F16 -DVW=1")
build_kernel(f16 axpy_fastest_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F16 -DVW=2")
build_kernel(f16 axpy_fastest_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F16 -DVW=4")
build_kernel(f16 axpy_fastest_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F16 -DVW=8")
build_kernel(f16 axpy_fastest_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F16 -DVW=16")

build_kernel(f32 axpy_fastest_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F32 -DVW=1")
build_kernel(f32 axpy_fastest_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F32 -DVW=2")
build_kernel(f32 axpy_fastest_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F32 -DVW=4")
build_kernel(f32 axpy_fastest_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F32 -DVW=8")
build_kernel(f32 axpy_fastest_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F32 -DVW=16")

build_kernel(f64 axpy_fastest_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F64 -DVW=1")
build_kernel(f64 axpy_fastest_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F64 -DVW=2")
build_kernel(f64 axpy_fastest_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F64 -DVW=4")
build_kernel(f64 axpy_fastest_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F64 -DVW=8")
build_kernel(f64 axpy_fastest_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F64 -DVW=16")

build_kernel(c64 axpy_fastest_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C64 -DVW=1")
build_kernel(c64 axpy_fastest_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C64 -DVW=2")
build_kernel(c64 axpy_fastest_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C64 -DVW=4")
build_kernel(c64 axpy_fastest_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C64 -DVW=8")
build_kernel(c64 axpy_fastest_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C64 -DVW=16")

build_kernel(c128 axpy_fastest_vw1 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C128 -DVW=1")
build_kernel(c128 axpy_fastest_vw2 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C128 -DVW=2")
build_kernel(c128 axpy_fastest_vw4 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C128 -DVW=4")
build_kernel(c128 axpy_fastest_vw8 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C128 -DVW=8")
build_kernel(c128 axpy_fastest_vw16 "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C128 -DVW=16")


set(ALL_KERNEL_NAMESPACES f16 f32 f64 c64 c128)

