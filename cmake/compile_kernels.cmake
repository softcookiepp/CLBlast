

# most args will be consistent, so wrap it
macro(build_kernel KERNEL_NAMESPACE KERNEL_NAME INPUT_FILENAME BUILD_ARGS)
	message(${BUILD_ARGS})
	tart_build_kernel_spv(${KERNEL_NAMESPACE} ${KERNEL_NAME} main glsl ${INPUT_FILENAME} "${BUILD_ARGS}")
endmacro()

# configure all kernels with variable vector with
set(VECTOR_WIDTHS 1 2 4 8 16)
set(VW_KERNELS_LEVEL1
	axpy axpy_faster axpy_fastest
	copy copy_fast
	scal scal_fast
	swap swap_fast)
foreach(KERNEL ${VW_KERNELS_LEVEL1})
	foreach(VW ${VECTOR_WIDTHS})
		build_kernel(f16 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F16 -DVW=${VW}")
		build_kernel(f32 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F32 -DVW=${VW}")
		build_kernel(f64 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F64 -DVW=${VW}")
		build_kernel(c64 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C64 -DVW=${VW}")
		build_kernel(c128 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C128 -DVW=${VW}")
	endforeach()
endforeach()

set(KERNELS_LEVEL2
	gemv ger her her2 trsv_forward trsv_backward trsv_fill_vector)
foreach(KERNEL ${KERNELS_LEVEL2})
		build_kernel(f16 "${KERNEL}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F16")
		build_kernel(f32 "${KERNEL}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F32")
		build_kernel(f64 "${KERNEL}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F64")
		build_kernel(c64 "${KERNEL}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C64")
		build_kernel(c128 "${KERNEL}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C128")
endforeach()

set(VW_KERNELS_LEVEL2
	gemv_fast gemv_fast_rot)
foreach(KERNEL ${VW_KERNELS_LEVEL2})
	foreach(VW ${VECTOR_WIDTHS})
		build_kernel(f16 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F16 -DVW=${VW}")
		build_kernel(f32 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F32 -DVW=${VW}")
		build_kernel(f64 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F64 -DVW=${VW}")
		build_kernel(c64 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C64 -DVW=${VW}")
		build_kernel(c128 "${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C128 -DVW=${VW}")
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

set(NO_FLAG_KERNELS gemv)

set(ALL_KERNEL_NAMESPACES f16 f32 f64 c64 c128)

