//#version 450
//#include "../common.glsl"
//#define ROUTINE_TRSV 1
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains kernels to perform forward or backward substition, as used in the TRSV routine
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_TRSV)

#if USE_BDA
#else
	layout(binding = 0, std430) writeonly buffer dest_buf { real dest[]; };
#endif

layout(push_constant) uniform FillVector
{
	int n;
	int inc;
	int offset;
#if USE_BDA
	__global real* restrict dest;
#endif
	real_arg arg_value;
} args;

void main()
{
	const real value = GetRealArg(args.arg_value);
	const int tid = get_global_id(0);
	if (tid < args.n) {
		dest[tid*args.inc + args.offset] = value;
	}
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
