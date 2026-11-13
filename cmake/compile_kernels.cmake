

# most args will be consistent, so wrap it
macro(build_kernel KERNEL_NAMESPACE KERNEL_NAME INPUT_FILENAME BUILD_ARGS)
	message(${BUILD_ARGS})
	tart_build_kernel_spv(${KERNEL_NAMESPACE} ${KERNEL_NAME} main glsl ${INPUT_FILENAME} "${BUILD_ARGS}")
endmacro()

# used by several configuration loops
set(BINARY_FLAG_OPTIONS 0 1)
set(VECTOR_WIDTHS 1 2 4 8 16)

set(LEVEL3_DUAL_VECTOR_WIDTH_KERNELS gemm_direct gemm_direct_batched gemm_direct_strided_batched)
foreach(KERNEL ${LEVEL3_DUAL_VECTOR_WIDTH_KERNELS})
	foreach(VWM ${VECTOR_WIDTHS})
		foreach(VWN ${VECTOR_WIDTHS})
			build_kernel(f16 "${KERNEL}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F16 -DVWM=${VWM} -DVWN=${VWN}")
			build_kernel(f32 "${KERNEL}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F32 -DVWM=${VWM} -DVWN=${VWN}")
			build_kernel(f64 "${KERNEL}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F64 -DVWM=${VWM} -DVWN=${VWN}")
			build_kernel(c64 "${KERNEL}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C64 -DVWM=${VWM} -DVWN=${VWN}")
			build_kernel(c128 "${KERNEL}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C128 -DVWM=${VWM} -DVWN=${VWN}")
		endforeach()
	endforeach()
endforeach()

set(GEMM_VW 1)
set(LEVEL3_SUBGROUP_DUAL_VECTOR_WIDTH_KERNELS gemm gemm_upper gemm_lower gemm_batched gemm_strided_batched)
foreach(KERNEL ${LEVEL3_SUBGROUP_DUAL_VECTOR_WIDTH_KERNELS})
	foreach(SUBGROUP_SUPPORT ${BINARY_FLAG_OPTIONS})
		foreach(VWM ${GEMM_VW})
			foreach(VWN ${GEMM_VW})
				build_kernel(f16 "${KERNEL}_sub${SUBGROUP_SUPPORT}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F16 -DVWM=${VWM} -DVWN=${VWN} -DSUBGROUP_OPERATIONS_SUPPORTED=${SUBGROUP_SUPPORT}")
				build_kernel(f32 "${KERNEL}_sub${SUBGROUP_SUPPORT}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F32 -DVWM=${VWM} -DVWN=${VWN} -DSUBGROUP_OPERATIONS_SUPPORTED=${SUBGROUP_SUPPORT}")
				build_kernel(f64 "${KERNEL}_sub${SUBGROUP_SUPPORT}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/x${KERNEL}.glsl" "-DDTYPE=DTYPE_F64 -DVWM=${VWM} -DVWN=${VWN} -DSUBGROUP_OPERATIONS_SUPPORTED=${SUBGROUP_SUPPORT}")
				build_kernel(c64 "${KERNEL}_sub${SUBGROUP_SUPPORT}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C64 -DVWM=${VWM} -DVWN=${VWN} -DSUBGROUP_OPERATIONS_SUPPORTED=${SUBGROUP_SUPPORT}")
				build_kernel(c128 "${KERNEL}_sub${SUBGROUP_SUPPORT}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/x${KERNEL}.glsl" "-DDTYPE=DTYPE_C128 -DVWM=${VWM} -DVWN=${VWN} -DSUBGROUP_OPERATIONS_SUPPORTED=${SUBGROUP_SUPPORT}")
			endforeach()
		endforeach()
	endforeach()
endforeach()

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

set(ALL_KERNEL_NAMESPACES f16 f32 f64 c64 c128)

