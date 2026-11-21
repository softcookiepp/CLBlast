

# most args will be consistent, so wrap it
macro(build_kernel KERNEL_NAMESPACE KERNEL_NAME INPUT_FILENAME BUILD_ARGS)
	message(${BUILD_ARGS})
	tart_build_kernel_spv(${KERNEL_NAMESPACE} ${KERNEL_NAME} main glsl ${INPUT_FILENAME} "${BUILD_ARGS}")
endmacro()

# used by several configuration loops
set(BINARY_FLAG_OPTIONS 0 1)
set(VECTOR_WIDTHS 1 2 4 8 16)
set(DTYPE_PREFIXES F16 F32 F64 C64 C128)

foreach(DTYPE_PREFIX ${DTYPE_PREFIXES})
	set(LEVELX_KERNELS im2col_kernel_normal col2im_kernel_normal)
	foreach(KERNEL ${LEVELX_KERNELS})
		build_kernel(blas "${DTYPE_PREFIX}${KERNEL}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/levelx/${KERNEL}.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX}")
	endforeach()

	set(LEVELX_CONVGEMM_KERNELS convgemm_kernel convgemm_normal)
	foreach(KERNEL ${LEVELX_CONVGEMM_KERNELS})
		foreach(VWM ${VECTOR_WIDTHS})
			foreach(VWN ${VECTOR_WIDTHS})
				build_kernel(blas "${DTYPE_PREFIX}${KERNEL}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/levelx/${KERNEL}.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX} -DVWMD=${VWM} -DVWND=${VWN}")
			endforeach()
		endforeach()
	endforeach()

	# only transpose_fast uses TRA_WPT, no other kernels use it
	foreach(VW ${VECTOR_WIDTHS})
		build_kernel(blas "${DTYPE_PREFIX}transpose_fast_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/transpose_fast.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX} -DTRA_WPT=${VW}")
	endforeach()


	set(LEVEL3_NOARG_NOX_KERNELS
		convert_hermitian_upper convert_hermitian_lower
		convert_symmetric_upper convert_symmetric_lower
		convert_triangular_upper convert_triangular_lower
		level3_fill_matrix
		transpose_pad_matrix
		transpose_pad_matrix_batched
		transpose_pad_matrix_strided_batched
		transpose_matrix
		transpose_matrix_batched
		transpose_matrix_strided_batched
		copy_matrix_fast
		copy_pad_matrix
		copy_pad_matrix_batched
		copy_pad_matrix_strided_batched
		copy_matrix
		copy_matrix_batched
		copy_matrix_strided_batched
		invert_diagonal_block
		invert_triple_matmul
		invert_triple_matmul_part2
		)
	foreach(KERNEL ${LEVEL3_NOARG_NOX_KERNELS})
		build_kernel(blas "${DTYPE_PREFIX}${KERNEL}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/${KERNEL}.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX}")
	endforeach()

	set(LEVEL3_DUAL_VECTOR_WIDTH_KERNELS xgemm_direct xgemm_direct_batched xgemm_direct_strided_batched)
	foreach(KERNEL ${LEVEL3_DUAL_VECTOR_WIDTH_KERNELS})
		foreach(VWM ${VECTOR_WIDTHS})
			foreach(VWN ${VECTOR_WIDTHS})
				build_kernel(blas "${DTYPE_PREFIX}${KERNEL}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/${KERNEL}.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX} -DVWM=${VWM} -DVWN=${VWN}")
			endforeach()
		endforeach()
	endforeach()

	set(GEMM_VW 1)
	set(LEVEL3_SUBGROUP_DUAL_VECTOR_WIDTH_KERNELS xgemm xgemm_upper xgemm_lower xgemm_batched xgemm_strided_batched)
	foreach(KERNEL ${LEVEL3_SUBGROUP_DUAL_VECTOR_WIDTH_KERNELS})
		foreach(SUBGROUP_SUPPORT ${BINARY_FLAG_OPTIONS})
			foreach(VWM ${GEMM_VW})
				foreach(VWN ${GEMM_VW})
					build_kernel(blas "${DTYPE_PREFIX}${KERNEL}_sub${SUBGROUP_SUPPORT}_vw${VWM}_vw${VWN}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/${KERNEL}.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX} -DVWM=${VWM} -DVWN=${VWN} -DSUBGROUP_OPERATIONS_SUPPORTED=${SUBGROUP_SUPPORT}")
				endforeach()
			endforeach()
		endforeach()
	endforeach()

	set(VW_KERNELS_LEVEL1
		xaxpy xaxpy_faster xaxpy_fastest
		xcopy xcopy_fast
		xscal xscal_fast
		xswap xswap_fast
		xhad xhad_faster xhad_fastest)
	foreach(KERNEL ${VW_KERNELS_LEVEL1})
		foreach(VW ${VECTOR_WIDTHS})
			build_kernel(blas "${DTYPE_PREFIX}${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/${KERNEL}.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX} -DVW=${VW}")
		endforeach()
	endforeach()

	set(KERNELS_LEVEL1
		xaxpy_batched)
	foreach(KERNEL ${KERNELS_LEVEL1})
		build_kernel(blas "${DTYPE_PREFIX}${KERNEL}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/${KERNEL}.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX}")
	endforeach()

	set(KERNELS_LEVEL2
		xgemv xger xher xher2 xtrsv_forward xtrsv_backward xtrsv_fill_vector)
	foreach(KERNEL ${KERNELS_LEVEL2})
			build_kernel(blas "${DTYPE_PREFIX}${KERNEL}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/${KERNEL}.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX}")
	endforeach()

	set(VW_KERNELS_LEVEL2
		xgemv_fast xgemv_fast_rot)
	foreach(KERNEL ${VW_KERNELS_LEVEL2})
		foreach(VW ${VECTOR_WIDTHS})
			build_kernel(blas "${DTYPE_PREFIX}${KERNEL}_vw${VW}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/${KERNEL}.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX} -DVW=${VW}")
		endforeach()
	endforeach()


	set(REDUCE_KERNELS
		xamax xamax_epilogue
		xasum xasum_epilogue
		xdot xdot_epilogue
		xnrm2 xnrm2_epilogue)
	foreach(KERNEL ${REDUCE_KERNELS})
		build_kernel(blas "${DTYPE_PREFIX}${KERNEL}" "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/${KERNEL}.glsl" "-DDTYPE=DTYPE_${DTYPE_PREFIX}")
	endforeach()
endforeach()
