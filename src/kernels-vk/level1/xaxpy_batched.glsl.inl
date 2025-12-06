// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xaxpy kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=1) and no offsets (offx=offy=0). Another version is more general, but doesn't
// support vector data-types. The general version has a batched implementation as well.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Full version of the kernel with offsets and strided accesses: batched version
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif
layout(push_constant) uniform XaxpyBatched(const int n, const __constant real_arg* arg_alphas,
									const __global real* restrict xgm, const __constant int* x_offsets, const int x_inc,
									__global real* ygm, const __constant int* y_offsets, const int y_inc)
void main()
{
	const int batch = get_group_id(1);
	const real alpha = GetRealArg(arg_alphas[batch]);

	// Loops over the work that needs to be done (allows for an arbitrary number of threads)
	for (int id = get_global_id(0); id < n; id += get_global_size(0))
	{
		// will have to update this later
		realV xvalue = xgm[id*x_inc + x_offsets[batch]];
		MultiplyAdd(ygm[id*y_inc + y_offsets[batch]], alpha, xvalue);
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
