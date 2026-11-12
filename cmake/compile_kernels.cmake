

# most args will be consistent, so wrap it
macro(build_kernel KERNEL_NAMESPACE KERNEL_NAME INPUT_FILENAME BUILD_ARGS)
	message(${BUILD_ARGS})
	tart_build_kernel_spv(${KERNEL_NAMESPACE} ${KERNEL_NAME} main glsl ${INPUT_FILENAME} "${BUILD_ARGS}")
endmacro()

# configure all kernels with variable vector with
set(VECTOR_WIDTHS 1 2 4 8 16)
foreach(VW ${VECTOR_WIDTHS})
	# axpy
	build_kernel(f16 "axpy_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F16 -DVW=${VW}")
	build_kernel(f32 "axpy_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F32 -DVW=${VW}")
	build_kernel(f64 "axpy_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_F64 -DVW=${VW}")
	build_kernel(c64 "axpy_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C64 -DVW=${VW}")
	build_kernel(c128 "axpy_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy.glsl" "-DDTYPE=DTYPE_C128 -DVW=${VW}")
	
	#axpy faster
	build_kernel(f16 "axpy_faster_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F16 -DVW=${VW}")
	build_kernel(f32 "axpy_faster_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F32 -DVW=${VW}")
	build_kernel(f64 "axpy_faster_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_F64 -DVW=${VW}")
	build_kernel(c64 "axpy_faster_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C64 -DVW=${VW}")
	build_kernel(c128 "axpy_faster_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_faster.glsl" "-DDTYPE=DTYPE_C128 -DVW=${VW}")
	
	#axpy fastest
	build_kernel(f16 "axpy_fastest_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F16 -DVW=${VW}")
	build_kernel(f32 "axpy_fastest_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F32 -DVW=${VW}")
	build_kernel(f64 "axpy_fastest_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_F64 -DVW=${VW}")
	build_kernel(c64 "axpy_fastest_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C64 -DVW=${VW}")
	build_kernel(c128 "axpy_fastest_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/xaxpy_fastest.glsl" "-DDTYPE=DTYPE_C128 -DVW=${VW}")
endforeach()

set(ELEMENTWISE_2SPEED_KERNELS copy scal swap)
foreach(KERNEL ${ELEMENTWISE_2SPEED_KERNELS})
	foreach(VW ${VECTOR_WIDTHS})
		build_kernel(f16 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F16 -DVW=${VW}")
		build_kernel(f32 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F32 -DVW=${VW}")
		build_kernel(f64 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F64 -DVW=${VW}")
		build_kernel(c64 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C64 -DVW=${VW}")
		build_kernel(c128 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C128 -DVW=${VW}")
		
		build_kernel(f16 "${KERNEL}_fast_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_fast.glsl" "-DDTYPE=DTYPE_F16 -DVW=${VW}")
		build_kernel(f32 "${KERNEL}_fast_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_fast.glsl" "-DDTYPE=DTYPE_F32 -DVW=${VW}")
		build_kernel(f64 "${KERNEL}_fast_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_fast.glsl" "-DDTYPE=DTYPE_F64 -DVW=${VW}")
		build_kernel(c64 "${KERNEL}_fast_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_fast.glsl" "-DDTYPE=DTYPE_C64 -DVW=${VW}")
		build_kernel(c128 "${KERNEL}_fast_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_fast.glsl" "-DDTYPE=DTYPE_C128 -DVW=${VW}")
	endforeach()
endforeach()


set(REDUCE_KERNELS amax asum dot nrm2)
foreach(KERNEL ${REDUCE_KERNELS})
	build_kernel(f16 ${KERNEL} "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F16")
	build_kernel(f32 ${KERNEL} "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F32")
	build_kernel(f64 ${KERNEL} "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F64")
	build_kernel(c64 ${KERNEL} "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C64")
	build_kernel(c128 ${KERNEL} "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C128")
	build_kernel(f16 "${KERNEL}_epilogue" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_epilogue.glsl" "-DDTYPE=DTYPE_F16")
	build_kernel(f32 "${KERNEL}_epilogue" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_epilogue.glsl" "-DDTYPE=DTYPE_F32")
	build_kernel(f64 "${KERNEL}_epilogue" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_epilogue.glsl" "-DDTYPE=DTYPE_F64")
	build_kernel(c64 "${KERNEL}_epilogue" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_epilogue.glsl" "-DDTYPE=DTYPE_C64")
	build_kernel(c128 "${KERNEL}_epilogue" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}_epilogue.glsl" "-DDTYPE=DTYPE_C128")
endforeach()

set(ALL_KERNEL_NAMESPACES f16 f32 f64 c64 c128)

