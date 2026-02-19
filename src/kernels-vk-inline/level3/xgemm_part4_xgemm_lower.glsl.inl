
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 4 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef XGEMM_PART3_GLSL
#define XGEMM_PART3_GLSL


// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef XGEMM_PART2_GLSL
#define XGEMM_PART2_GLSL


// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains two optimized matrix-multiplication kernels:
// - Kernel 0: inspired by the paper by Matsumoto et al. and the tutorial on
//   http://www.cedricnugteren.nl/tutorial.php
// - Kernel 1: inspired by a Qualcomm optimized GPU kernel with 2D register tiling
//   https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
// Both are fully configurable (and tunable!) using many parameters. Both kernels support
// different data-types (SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM) through a pre-processor define.
//
// For kernel 0 matrices are accessed as follows:
// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
// For kernel 1, both A and C are transposed w.r.t. the above
//
// Or as an image (assuming column-major)
//       K                      
//    o-------o                 
//    |       |                 
//  N | [B^T] |                 
//    |       |                 
//    o-------o                 
//        K               N     
//    o-------o        o-----o  
//  M |  [A]  |      M | [C] |  
//    |       |        |     |  
//    o-------o        o-----o  
//                              
//
// This kernel is separated into multiple files. This is part 1 out of 4.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef XGEMM_PART1_GLSL
#define XGEMM_PART1_GLSL


// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef COMMON_GLSL
#define COMMON_GLSL
// =================================================================================================

// whether or not to use buffer device addresses instead of descriptors
// default to false
#ifndef USE_BDA
	#define USE_BDA 0
#endif

// if 64-bit integers are supported
#ifndef USE_INT64
	#define USE_INT64 0
#endif

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this file is used outside of the CLBlast library.
#ifndef PRECISION
	#define PRECISION 32			// Data-types: half, single or double precision, complex or regular
#endif

// =================================================================================================
	
// reserved for when unrolling semantics are able to be used
#ifndef UNROLL
	#define UNROLL(N)
#endif

// support for subgroup operations
#ifndef USE_SUBGROUP_SHUFFLING
	#define USE_SUBGROUP_SHUFFLING 0
#endif
#if USE_SUBGROUP_SHUFFLING
	//#extension GL_EXT_shader_subgroup : require
	#extension GL_KHR_shader_subgroup_shuffle : require
#endif

// Enable support for half-precision
#if PRECISION == 16
	#extension GL_EXT_shader_16bit_storage : require
	#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

// Enable support for double-precision
#if PRECISION == 64 || PRECISION == 6464
	#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#endif

#ifdef USE_INT64
	#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#endif

#if USE_BDA
	// if BDA is supported but uint64_t isn't, then we need to use uvec2 for the address types
	#if USE_UINT64
		#extension GL_EXT_buffer_reference : require
		#define addr_t uint64_t
		uint64_t addPtrOffset(uint64_t ptr, uint64_t offset) { return ptr + offset; }
		uint64_t addPtrOffset(uint64_t ptr, uint offset) { return ptr + uint64_t(offset); }
		uint64_t mulPtrCoef(uint64_t ptr, uint64_t coef) { return ptr*coef; }
		uint64_t mulPtrCoef(uint64_t ptr, uint coef) { return ptr*uint64_t(coef); }
	#else
		#extension GL_EXT_buffer_reference_uvec2 : require
		#define addr_t uvec2
		uvec2 addPtrOffset(uvec2 addr, uvec2 offset)
		{
			uint carry;
			uint lo = uaddCarry(addr.x, offset.x, carry);
			uint hi = addr.y + offset.y + carry;
			return uvec2(lo, hi);
		}
		uvec2 addPtrOffset(uvec2 addr, uint offset)
		{
			uint carry;
			uint lo = uaddCarry(addr.x, offset, carry);
			uint hi = addr.y + carry;
			return uvec2(lo, hi);
		}
	#endif
#endif


// Half-precision
#if PRECISION == 16
	struct vec2_t { f16vec2 s; };
	struct vec4_t { f16vec4 s; };
	struct vec8_t { float16_t s[8]; };
	struct vec16_t { float16_t s[16]; };
	#define DTYPE_SIZE 2
	#define real float16_t
	#define ZERO float16_t(0.0)
	#define ONE float16_t(1.0)
	#define SMALLEST -1.0e14
	#define INFINITY float16_t(uintBitsToFloat(0x7F800000))
	#define NAN float16_t(uintBitsToFloat(0x7FC00000))
	#define PI float16_t(3.14159265358979323846)

// Single-precision
#elif PRECISION == 32
	struct vec2_t { vec2 s; };
	struct vec4_t { vec4 s; };
	struct vec8_t { float s[8]; };
	struct vec16_t { float s[16]; };
	#define DTYPE_SIZE 4
	#define real float
	#define ZERO real(0.0f)
	#define ONE 1.0f
	#define SMALLEST -1.0e37f
	#define INFINITY uintBitsToFloat(0x7F800000)
	#define NAN uintBitsToFloat(0x7FC00000)
	#define PI float(3.14159265358979323846)

// Double-precision 
#elif PRECISION == 64
	struct vec2_t { dvec2 s; };
	struct vec4_t { dvec4 s; };
	struct vec8_t { double s[8]; };
	struct vec16_t { double s[16]; };
	#define DTYPE_SIZE 8
	#define real double
	#define ZERO 0.0
	#define ONE 1.0
	#define SMALLEST -1.0e37
	#define INFINITY double(uintBitsToFloat(0x7F800000))
	#define NAN double(uintBitsToFloat(0x7FC00000))
	#define PI double(3.14159265358979323846)

// Complex single-precision
#elif PRECISION == 3232
	struct vec2_t { mat2x2 s; };
	struct vec4_t { mat4x2 s; };
	struct vec8_t { vec2 s[8]; };
	struct vec16_t { vec2 s[16]; };
	#define DTYPE_SIZE 8
	#define real vec2
	#define ZERO 0.0f
	#define ONE 1.0f
	#define SMALLEST -1.0e37f
	#define INFINITY uintBitsToFloat(0x7F800000)
	#define NAN uintBitsToFloat(0x7FC00000)
	#define PI float(3.14159265358979323846)

// Complex double-precision
#elif PRECISION == 6464
	struct vec2_t { dmat2x2 s; };
	struct vec4_t { dmat4x2 s; };					 
	struct vec8_t { dvec2 s[8]; };
	struct vec16_t { dvec2 s[16]; };
	#define DTYPE_SIZE 16
	#define real dvec2
	#define ZERO 0.0
	#define ONE 1.0
	#define SMALLEST -1.0e37
	#define INFINITY double(uintBitsToFloat(0x7F800000))
	#define NAN double(uintBitsToFloat(0x7FC00000))
	#define PI double(3.14159265358979323846)
#endif

// this simplifies stuff c:
#define real2 vec2_t
#define real4 vec4_t
#define real8 vec8_t
#define real16 vec16_t

#if USE_BDA
	// floating point BDA
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE) buffer real_ptr_t { real s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*2) buffer real2_ptr_t { vec2_t s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*4) buffer real4_ptr_t { real4 s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*8) buffer real8_ptr_t { real8 s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*16) buffer real16_ptr_t { real16 s[]; };
	
	// uint BDA
	layout(buffer_reference, buffer_reference_align = 4) buffer int_ptr_t { int s[]; };
	layout(buffer_reference, buffer_reference_align = 4) buffer uint_ptr_t { uint s[]; };
	#if 0
		// index function to make BDA and non-BDA devices share the same code
		#define INDEX_AS_ALIGNED(buffer_t, ptr, index, alignment) buffer_t(addPtrOffset(addr_t(ptr), index*alignment))
		real_ptr_t indexGMimpl(real_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real_ptr_t, ptr, index, DTYPE_SIZE); }
		real2_ptr_t indexGMimpl(real2_ptr_t ptr, uint index, uint gae) { return INDEX_AS_ALIGNED(real2_ptr_t, ptr, index, DTYPE_SIZE*2); }
		real4_ptr_t indexGMimpl(real4_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real4_ptr_t, ptr, index, DTYPE_SIZE*4); }
		real8_ptr_t indexGMimpl(real8_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real8_ptr_t, ptr, index, DTYPE_SIZE*8); }
		real16_ptr_t indexGMimpl(real16_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real16_ptr_t, ptr, index, DTYPE_SIZE*16); }
		int_ptr_t indexGMimpl(int_ptr_t, uint index) {return INDEX_AS_ALIGNED(int_ptr_t, ptr, index, 4); }
		uint_ptr_t indexGMimpl(uint_ptr_t, uint index) {return INDEX_AS_ALIGNED(uint_ptr_t, ptr, index, 4); }
	
		#define indexGM(ptr, index) indexGMimpl(ptr, index).s[0]
	#else
		// that, above, does not seem to work right now.
		// Therefore, it is only possible to do it this way:
		#define indexGM(ptr, index) ptr.s[index]
	#endif
#else
	#define indexGM(ptr, index) ptr[index]
#endif

// Single-element version of a complex number
#if PRECISION == 3232
	#define singlereal float
#elif PRECISION == 6464
	#define singlereal double 
#else
	#if USE_BDA
		#define singlereal_ptr_t real_ptr_t
	#endif
	#define singlereal real 
#endif

#if USE_BDA && PRECISION >= 3232
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE/2) buffer singlereal_ptr_t { singlereal s[]; };
	//
#endif

// Converts a 'real argument' value to a 'real' value as passed to the kernel. Normally there is no
// conversion, but half-precision is not supported as kernel argument so it is converted from float.
#if PRECISION == 16
	#define real_arg float 
	#define GetRealArg(x) float16_t(x)
#else
	#define real_arg real 
	#define GetRealArg(x) real(x)
#endif

//#elif PRECISION == 64
//	// lets see if this makes our life easier...
//	#define real_arg float
//	#define GetRealArg(x) double(x)

// Pointers to local memory objects (using a define because CUDA doesn't need them)
#ifndef LOCAL_PTR
	#define LOCAL_PTR shared
#endif

// =================================================================================================

// Don't use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific
// devices, this is enabled (see src/routine.cpp).
#ifndef USE_CL_MAD
	#define USE_CL_MAD 0
#endif

// By default the workgroup size requirement is enabled. For Qualcomm devices the workgroup size 
// requirement results in worse performance and is disabled (src/utilities/compile.cpp)
#ifndef RELAX_WORKGROUP_SIZE
	#define RELAX_WORKGROUP_SIZE 0
#endif

// ensure all spec constants related to workgroup size are here and ready
#if RELAX_WORKGROUP_SIZE
	layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
#endif

// Sets a variable to zero
#if PRECISION == 3232 || PRECISION == 6464
	#define SetToZero(a) a = real(ZERO, ZERO)
#else
	#define SetToZero(a) a = ZERO
#endif

// Sets a variable to zero (only the imaginary part)
#if PRECISION == 3232 || PRECISION == 6464
	#define ImagToZero(a) a.y = ZERO
#else
	#define ImagToZero(a) 
#endif

// Sets a variable to one
#if PRECISION == 3232 || PRECISION == 6464
	#define SetToOne(a) a = real(ONE, ZERO)
#else
	#define SetToOne(a) a = ONE
#endif

// Determines whether a variable is zero
#if PRECISION == 3232 || PRECISION == 6464
	#define IsZero(a) ((a[0] == ZERO) && (a[1] == ZERO))
#else
	#define IsZero(a) (a == ZERO)
#endif

// component-wise absolute value
#define AbsoluteValue(value) value = abs(value)

// Negation (component-wise)
#if 0 //PRECISION == 3232 || PRECISION == 6464
	#define Negate(value) value.x = (-1.0(value.x)); value.y = (-1.0*(value.y))
#else
	#define Negate(value) value = (-1.0*(value))
#endif

// Adds two complex variables
#if 0 //PRECISION == 3232 || PRECISION == 6464
	#define Add(c,a,b) c = real(a.x + b.x, a.y + b.y)
#else
	#define Add(c,a,b) c = a + b
#endif

// Subtracts two complex variables
#if 0 //PRECISION == 3232 || PRECISION == 6464
	#define Subtract(c,a,b) c = real(a.x - b.x, a.y - b.y)
#else
	#define Subtract(c,a,b) c = a - b
#endif

// Multiply two complex variables (used in the defines below)
#if PRECISION == 3232 || PRECISION == 6464
	#define MulReal(a,b) a[0]*b[0] - a[1]*b[1]
	#define MulImag(a,b) a[0]*b[1] + a[1]*b[0]
#endif

// The scalar multiply function
#if PRECISION == 3232 || PRECISION == 6464
	#define Multiply(c,a,b) c = real(MulReal(a,b), MulImag(a,b))
#else
	#define Multiply(c,a,b) c = a * b
#endif

#define vMultiply(c, a, b, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		Multiply(c.s[i], a.s[i], b.s[i]); \
	} \
}

// c is vector, a is scalar, b is vector
#define vsMultiply(c, a, b, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		Multiply(c.s[i], a, b.s[i]); \
	} \
}

// c is vector, a is scalar, b is vector
#define vsMultiplyAdd(c, a, b, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		MultiplyAdd(c.s[i], a, b.s[i]); \
	} \
}

#define vSetToZero(v, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		SetToZero(v.s[i]); \
	} \
}

// The scalar multiply-add function
#if PRECISION == 3232 || PRECISION == 6464
	#define MultiplyAdd(c,a,b) c += real(MulReal(a,b), MulImag(a,b))
#else
	#if USE_CL_MAD == 1
		#define MultiplyAdd(c,a,b) c = fma(a, b, c)
	#else
		#define MultiplyAdd(c,a,b) c += (a * b)
	#endif
#endif

// The scalar multiply-subtract function
#if PRECISION == 3232 || PRECISION == 6464
	#define MultiplySubtract(c,a,b) c -= real(MulReal(a,b), MulImag(a,b))
#else
	#define MultiplySubtract(c,a,b) c -= (a * b)
#endif

// The scalar division function: full division
#if PRECISION == 3232 || PRECISION == 6464
	#define DivideFull(c,a,b) singlereal num_x = (a.x * b.x) + (a.y * b.y); singlereal num_y = (a.y * b.x) - (a.x * b.y); singlereal denom = (b.x * b.x) + (b.y * b.y); c = real(num_x / denom, num_y / denom)
#elif PRECISION == 16
	// Some hardware doesn't compute NaN properly
	// #define DivideFull(c,a,b) c = (b == ZERO ? NAN : a / b)
	// still need to test the specifics.
	// until then, just use default behavior...
	#define DivideFull(c,a,b) c = a / b
#else
	#define DivideFull(c,a,b) c = a / b
#endif

// The scalar AXPBY function
#if PRECISION == 3232 || PRECISION == 6464
	//#define AXPBY(e,a,b,c,d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
	#define AXPBY(e,a,b,c,d) e = real(MulReal(a,b) + MulReal(c,d), MulImag(a,b) + MulImag(c,d))
#else
	#define AXPBY(e,a,b,c,d) e = a*b + c*d
#endif

// The complex conjugate operation for complex transforms
#if PRECISION == 3232 || PRECISION == 6464
	#define COMPLEX_CONJUGATE(value) value.y = -1.0*value.y
#else
	#define COMPLEX_CONJUGATE(value) 
#endif

// =================================================================================================

// Macro for storing and loading, to accomodate BDA
#if USE_BDA
	// this needs to be changed, but I forget how it works
	#if 1
		#define INDEX(buf, idx) buf.s[idx]
	#else
		// normally you could just use buf.s[idx], but sometimes the alignment does not match the element size.
		#error "not implemented"
	#endif
#else
	#define INDEX(buf, idx) buf[idx]
#endif

// =================================================================================================

// vector load methods
#define vload2_single_alignment(index, buf) real2(INDEX(buf, index), INDEX(buf, index+1))
#define vload4_signle_alignment(index, buf) real4(INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2), INDEX(buf, index+3))

#define vload2(index, buf) real2(real[2](INDEX(buf, index), INDEX(buf, index+1)))
#define vload4(index, buf) real4(real[4](INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2), INDEX(buf, index+3)))
#define vload8(index, buf) real8(real[8](INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2),\
	INDEX(buf, index+3), INDEX(buf, index+4), INDEX(buf, index+5), INDEX(buf, index+6), INDEX(buf, index+7)))
#define vload16(index, buf) real16(real[16](INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2),\
	INDEX(buf, index+3), INDEX(buf, index+4), INDEX(buf, index+5), INDEX(buf, index+6), INDEX(buf, index+7),\
	INDEX(buf, index+8), INDEX(buf, index+9), INDEX(buf, index+10), INDEX(buf, index+11), INDEX(buf, index+12),\
	INDEX(buf, index+13), INDEX(buf, index+14), INDEX(buf, index+15) ))
	
#define vloadN(index, buf, N) vload##N(index, buf)

#define vTranspose(dst, src, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		UNROLL(vWidth) \
		for (uint j = 0; j < vWidth; j += 1) dst[i].s[j] = src[j].s[i]; \
	} \
}

// =================================================================================================

// Shuffled workgroup indices to avoid partition camping, see below. For specific devices, this is
// enabled (see src/routine.cc).
#ifndef USE_STAGGERED_INDICES
	#define USE_STAGGERED_INDICES 0
#endif

// because I am extremely lazy hehe
#define get_global_id(dim) int(gl_GlobalInvocationID[dim])
#define get_local_id(dim) int(gl_LocalInvocationID[dim])
#define get_group_id(dim) int(gl_WorkGroupID[dim])
#define get_global_size(idx) int(gl_NumWorkGroups[idx] * gl_WorkGroupSize[idx])
#define get_local_size(idx) int(gl_WorkGroupSize[idx])
#define get_num_groups(dim) int(gl_NumWorkGroups[dim])
#define get_sub_group_local_id() int(gl_SubgroupInvocationID)

// Staggered/shuffled group indices to avoid partition camping (AMD GPUs). Formula's are taken from:
// http://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf
// More details: https://github.com/CNugteren/CLBlast/issues/53
#if USE_STAGGERED_INDICES == 1 && GEMMK == 0
	int GetGroupIDFlat() {
		return get_group_id(0) + get_num_groups(0) * get_group_id(1);
		//return gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.y;
	}
	int GetGroupID1() {
		return (GetGroupIDFlat()) % get_num_groups(1);
		//return int((GetGroupIDFlat()) % gl_NumWorkGroups.y);
	}
	int GetGroupID0() {
		return ((GetGroupIDFlat() / get_num_groups(1)) + GetGroupID1()) % get_num_groups(0);
		//return int(((GetGroupIDFlat() / gl_NumWorkGroups.y) + GetGroupID1()) % gl_WorkGroupSize.x);
	}
#else
	int GetGroupID1() { return get_group_id(1); }
	//int GetGroupID1() { return int(gl_WorkGroupID.y); }
	int GetGroupID0() { return get_group_id(0); }
	//int GetGroupID0() { return int(gl_WorkGroupID.x); }
#endif

// =================================================================================================

// End of the C++11 raw string literal
#endif

// =================================================================================================


// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common functions and parameters specific for level 3 BLAS kernels.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef LEVEL3_GLSL
#define LEVEL3_GLSL
// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

// For the 'fast' copy kernel
#ifndef COPY_DIMX
	#define COPY_DIMX 8			// Local workgroup size in the first dimension (x)
#endif
#ifndef COPY_DIMY
	#define COPY_DIMY 8			// Local workgroup size in the second dimension (y)
#endif
#ifndef COPY_WPT
	#define COPY_WPT 1			 // Work per thread in the first dimension (x)
#endif
#ifndef COPY_VW
	#define COPY_VW 1				// Vector width in the second dimension (y)
#endif

// For the padding/copy kernels and the conversion kernels
#ifndef PAD_DIMX
	#define PAD_DIMX 8			// Local workgroup size in the first dimension (x)
#endif
#ifndef PAD_DIMY
	#define PAD_DIMY 8			// Local workgroup size in the second dimension (y)
#endif
#ifndef PAD_WPTX
	#define PAD_WPTX 1			// Work per thread in the first dimension (x)
#endif
#ifndef PAD_WPTY
	#define PAD_WPTY 1			// Work per thread in the second dimension (y)
#endif

// For the 'fast' transpose kernel
#ifndef TRA_DIM
	#define TRA_DIM 8			 // Number of local threads in the two dimensions (x,y)
#endif
#ifndef TRA_WPT
	#define TRA_WPT 1			 // Work per thread in one dimension and vector-width in the other
#endif
#ifndef TRA_PAD
	#define TRA_PAD 0			 // Padding of the local memory to avoid bank-conflicts
#endif
#ifndef TRA_SHUFFLE
	#define TRA_SHUFFLE 0	 // Shuffling of the global indices to avoid global memory bank-conflicts
#endif

// For the padding/transpose kernels
#ifndef PADTRA_TILE
	#define PADTRA_TILE 8	 // Number of local threads in the two dimensions (x,y)
#endif
#ifndef PADTRA_WPT
	#define PADTRA_WPT 1		// Amount of work per thread
#endif
#ifndef PADTRA_PAD
	#define PADTRA_PAD 0		// Padding of the local memory to avoid bank-conflicts
#endif

// =================================================================================================
#endif
// End of the C++11 raw string literal

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef GEMMK
	#define GEMMK 0		// Kernel to choose: 0 regular, 1 with 2D register tiling
#endif
#ifndef MWG
	#define MWG 8			// Tile-size in dimension M (e.g. 64, 128)
#endif
#ifndef NWG
	#define NWG 8			// Tile-size in dimension N (e.g. 64, 128)
#endif
#ifndef KWG
	#define KWG 8			// Tile-size in dimension K (e.g. 8, 16)
#endif
#ifndef MDIMC
	#define MDIMC 8		// Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMC
	#define NDIMC 8		// Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMA
	#define MDIMA 8		// Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)
#endif
#ifndef NDIMB
	#define NDIMB 8		// Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)
#endif
#ifndef KWI
	#define KWI 1			// Unroll factor of the KWG loop (smaller or equal than KWG)
#endif
#ifndef VWM
	#define VWM 1			// Vector width of matrices A and C
#endif
#ifndef VWN
	#define VWN 1			// Vector width of matrix B
#endif
#ifndef STRM
	#define STRM 0		 // Use strided access within a thread in the M-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef STRN
	#define STRN 0		 // Use strided access within a thread in the N-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef SA
	#define SA 0			 // Use local/shared memory to cache matrix A (1) or not (0) (kernel 0 only)
#endif
#ifndef SB
	#define SB 0			 // Use local/shared memory to cache matrix B (1) or not (0) (kernel 0 only)
#endif
#ifndef KREG
	#define KREG 1		 // Amount of register tiling in second dimension, multiple of VWN (kernel 1 only)
#endif

// Helper parameters based on the above tuning parameters
#define MWI (MWG/MDIMC)							 // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)							 // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)							 // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)							 // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)							 // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)							 // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#ifndef USE_VECTOR_MAD
	#define USE_VECTOR_MAD 0			// Unroll (0) or don't (1) unroll the vector MAD manually
#endif
#ifndef GLOBAL_MEM_FENCE
	#define GLOBAL_MEM_FENCE 0		// Global synchronisation barrier for potential better performance
#endif

#ifndef USE_SUBGROUP_SHUFFLING
	#define USE_SUBGROUP_SHUFFLING 0		 // Optionally enables subgroup shuffling for Intel GPUs
#endif

#if NWI != SUBGROUP_SIZE || MDIMC < SUBGROUP_SIZE
	#undef USE_SUBGROUP_SHUFFLING
	#define USE_SUBGROUP_SHUFFLING 0		 // Disables subgroups in case the assumptions don't hold
#endif

// =================================================================================================

// Data-widths in dimension M
#if VWM == 1
		#define realM real
#elif VWM == 2
		#define realM real2
#elif VWM == 4
		#define realM real4
#elif VWM == 8
		#define realM real8
#elif VWM == 16
		#define realM real16
#endif

// Data-widths in dimension N
#if VWN == 1
		#define realN real
#elif VWN == 2
		#define realN real2
#elif VWN == 4
		#define realN real4
#elif VWN == 8
		#define realN real8
#elif VWN == 16
		#define realN real16
#endif

// =================================================================================================

// Initializes the accumulation registers to zero
realM InitAccRegisters()
{
	realM result;
	#if VWM == 1
		SetToZero(result);
	#else
		vSetToZero(result, VWM);
	#endif
	return result;
}

// =================================================================================================

// buffer definitions (to avoid having to use macros everywhere like usual)
#if USE_BDA == 0
	layout(binding = 0, std430) buffer agm_buf { realM agm[]; };
	layout(binding = 1, std430) buffer bgm_buf { realN bgm[]; };
	layout(binding = 2, std430) buffer cgm_buf { realM cgm[]; };
	#if GEMMK == 1
		layout(binding = 3, std430) buffer agms_buf { real a_ptr[]; };
		layout(binding = 4, std430) buffer bgms_buf { real b_ptr[]; };
	#endif
#endif

// Allocates workgroup-private memory (local memory)
#if SA == 1
	shared realM alm[KWG * MWG/VWM];
#endif
#if SB == 1
	shared realN blm[KWG * NWG/VWN];
#endif

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
void GlobalToLocalA(
#if USE_BDA
	const __global realM* restrict agm,
#else
	int a_offset,
#endif
	//LOCAL_PTR realM* alm,
	const int kSizeM, const int tid, const int kwg)
{
	const int la0 = tid % MDIMA;
	const int la1 = tid / MDIMA;
	
	for (int _mia = 0; _mia < MWA/VWM; _mia += 1)
	{
		for (int _kia = 0; _kia < KWA; _kia += 1)
		{
			// Computes the indices based on strided/non-strided access
			#if STRM == 0
				int mg = _mia + la0*(MWA/VWM);
			#elif STRM == 1
				int mg = la0 + _mia*MDIMA;
			#endif

			// Computes the indices for the global memory
			int kg = _kia + la1*KWA;
			int idm = mg + GetGroupID0() * (MWG/VWM);
			int idk = kg + kwg;

			// Loads the data from global memory (not transposed) into the local memory
			alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm + a_offset];
		}
	}
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
void GlobalToLocalB(
#if USE_BDA
	const __global realN* restrict bgm,
#else
	int b_offset,
#endif
	// LOCAL_PTR realN* blm,
	const int kSizeN, const int tid, const int kwg)
{
	const int lb0 = tid % NDIMB;
	const int lb1 = tid / NDIMB;
	
	for (int _kib = 0; _kib < KWB; _kib += 1) {
		
		for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {

			// Computes the indices based on strided/non-strided access
			#if STRN == 0
				int ng = _nib + lb0*(NWB/VWN);
			#elif STRN == 1
				int ng = lb0 + _nib*NDIMB;
			#endif

			// Computes the indices for the global memory
			int kg = _kib + lb1*KWB;
			int idn = ng + GetGroupID1() * (NWG/VWN);
			int idk = kg + kwg;

			// Loads the data from global memory (transposed) into the local memory
			blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn + b_offset];
		}
	}
}
#endif

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0 && GEMMK == 0
realM GlobalToPrivateA(
#if USE_BDA
	const __global realM* restrict agm,
#else
	int a_offset,
#endif
	const int _mi, const int kSizeM, const int idk, const int kwg)
{
	// Computes the indices based on strided/non-strided access
	#if STRM == 0
		int mg = _mi + get_local_id(0)*(MWI/VWM);
	#elif STRM == 1
		int mg = get_local_id(0) + _mi*MDIMC;
	#endif

	// Computes the indices for the global memory
	int idm = mg + GetGroupID0() * (MWG/VWM);

	// Loads the data from global memory (not transposed) and stores into registers
	return agm[idk*(kSizeM/VWM) + idm + a_offset];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0 && GEMMK == 0
realN GlobalToPrivateB(
#if USE_BDA
	const __global realN* restrict bgm,
#else
	int b_offset,
#endif
	const int _ni, const int kSizeN, const int idk)
{
	// Computes the indices based on strided/non-strided access
	#if STRN == 0
		int ng = _ni + get_local_id(1)*(NWI/VWN);
	#elif STRN == 1
		int ng = get_local_id(1) + _ni*NDIMC;
	#endif

	// Computes the indices for the global memory
	int idn = ng + GetGroupID1() * (NWG/VWN);

	// Loads the data from global memory (transposed) and stores into registers
	return bgm[idk*(kSizeN/VWN) + idn + b_offset];
}
#endif

// =================================================================================================
#if GEMMK == 1

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix for kernel 1.
realN GlobalToPrivateA2D(
#if USE_BDA
	const __global real* restrict a_ptr,
#else
	int a_ptr_offset,
#endif
	const int tid_y, const int _ni, const int kSizeK, const int idk, const int _ki)
{
	#if PRECISION == 3232 || PRECISION == 6464
		const int a_index = (tid_y * NWI + _ni) * (kSizeK / VWN) + idk / VWN + _ki;
#if USE_BDA
		const __global realN* restrict agm = (const __global realN* restrict) a_ptr;
#endif
		// ok yeah, this is probably not going to work quite the way I thought it would...
		return agm[a_index];
	#else
		const int a_index = (tid_y * NWI + _ni) * kSizeK + idk + _ki * VWN + a_ptr_offset;
		#if VWN == 1
			return a_ptr[a_index];
		#else
			//return vload2(0, a_ptr + a_index);
			return vloadN(a_index, a_ptr, VWN);
		#endif
	#endif
}

// Same as above, but now for the B input matrix
realM GlobalToPrivateB2D(
#if USE_BDA
	const __global real* restrict b_ptr,
#else
	int b_ptr_offset,
#endif
	const int tid_x, const int _mi, const int kSizeN, const int idk, const int _ki)
{
	#if PRECISION == 3232 || PRECISION == 6464
		const int b_index = (idk + _ki) * (kSizeN / VWM) + tid_x * (MWI / VWM) + _mi;
#if USE_BDA
		const __global realM* restrict bgm = (const __global realM* restrict) b_ptr;
#endif
		// ok yeah, this is probably not going to work quite the way I thought it would...
		return bgm[b_index + b_ptr_offset/VWM];
	#else
		const int b_index = (idk + _ki) * kSizeN + tid_x * MWI + _mi * VWM + b_ptr_offset;
		#if VWM == 1
			return b_ptr[b_index];
		#elif VWM == 2
			//return vload2(0, b_ptr + b_index);
			return real2(b_ptr[b_index], b_ptr[b_index + 1]);
		#elif VWM == 4
			//return vload4(0, b_ptr + b_index);
			return real4(
				b_ptr[b_index],
				b_ptr[b_index + 1]
				b_ptr[b_index + 2]
				b_ptr[b_index + 3]
			);
		#elif VWM == 8
			//return vload8(0, b_ptr + b_index);
			return real8(
				real4(
					b_ptr[b_index],
					b_ptr[b_index + 1]
					b_ptr[b_index + 2]
					b_ptr[b_index + 3]
				),
				real4(
					b_ptr[b_index + 4],
					b_ptr[b_index + 5]
					b_ptr[b_index + 6]
					b_ptr[b_index + 7]
				)
			);
		#elif VWM == 16
			//return vload16(0, b_ptr + b_index);
			return real16(
				real4(
					b_ptr[b_index],
					b_ptr[b_index + 1]
					b_ptr[b_index + 2]
					b_ptr[b_index + 3]
				),
				real4(
					b_ptr[b_index + 4],
					b_ptr[b_index + 5]
					b_ptr[b_index + 6]
					b_ptr[b_index + 7]
				),
				real4(
					b_ptr[b_index + 8],
					b_ptr[b_index + 9]
					b_ptr[b_index + 10]
					b_ptr[b_index + 11]
				),
				real4(
					b_ptr[b_index + 12],
					b_ptr[b_index + 13]
					b_ptr[b_index + 14]
					b_ptr[b_index + 15]
				)
			);
		#endif
	#endif
}

#endif
// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
realM LocalToPrivateA(
	//LOCAL_PTR realM* alm,
	const int _mi, const int kg)
{
	#if STRM == 0
		int mg = _mi + get_local_id(0)*(MWI/VWM);
	#elif STRM == 1
		int mg = get_local_id(0) + _mi*MDIMC;
	#endif
	return alm[kg*(MWG/VWM) + mg];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
realN LocalToPrivateB(
	//LOCAL_PTR realN* blm,
	const int _ni, const int kg)
{
	#if STRN == 0
		int ng = _ni + get_local_id(1)*(NWI/VWN);
	#elif STRN == 1
		int ng = get_local_id(1) + _ni*NDIMC;
	#endif
	return blm[kg*(NWG/VWN) + ng];
}
#endif
#endif
// End of the C++11 raw string literal
// =================================================================================================


// The vectorised multiply-add function
realM MultiplyAddVector(realM cvec, const realM avec, const real bval) {
	#if USE_VECTOR_MAD == 1
		cvec += avec * bval;
	#else
		#if VWM == 1
			MultiplyAdd(cvec, avec, bval);
		#else
			vsMultiplyAdd(cvec, bval, avec, VWM);
		#endif
	#endif
	return cvec;
}

// =================================================================================================

// helper function since macro expressions don't like preprocessor conditions
ivec2 get_mg_ng_for_store(const int _mi, const int _ni)
{
	#if STRM == 0
		int mg = _mi + get_local_id(0)*(MWI/VWM);
	#elif STRM == 1
		int mg = get_local_id(0) + _mi*MDIMC;
	#endif
	#if STRN == 0
		int ng = _ni + get_local_id(1)*NWI;
	#elif STRN == 1
		int ng = _ni%VWN + get_local_id(1)*VWN + (_ni/VWN)*VWN*NDIMC;
	#endif
	return ivec2(mg, ng);
}

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
void StoreResults(
#if USE_BDA
		__global realM* cgm,
#else
		int c_offset,
#endif
		realM c_value, const int _mi, const int _ni,
		const int kSizeM, const real alpha, const real beta)
{
	ivec2 mgng = get_mg_ng_for_store(_mi, _ni);
	int mg = mgng[0];
	int ng = mgng[1];
	int idm = mg + GetGroupID0() * (MWG/VWM);
	int idn = ng + GetGroupID1() * NWG;
	int index = idn*(kSizeM/VWM) + idm;

	realM result;
	realM xval = c_value;

	// The final multiplication with alpha (in case beta == 0)
	if (IsZero(beta)) {
		#if VWM == 1
			Multiply(result, alpha, xval);
		#else
			vsMultiply(result, alpha, xval, VWM);
		#endif
	}

	// The final multiplication with alpha and the addition with beta*C
	else {
		realM yval = cgm[index + c_offset];
		#if VWM == 1
			AXPBY(result, alpha, xval, beta, yval);
		#else
			// TODO: make a macro for this that vectorizes stuff better. maybe.
			UNROLL(VWM)
			for (uint iv = 0; iv < VWM; iv += 1)
				AXPBY(result.s[iv], alpha, xval.s[iv], beta, yval.s[iv]);
		#endif
	}
	cgm[index + c_offset] = result;
}

#endif
// End of the C++11 raw string literal

// =================================================================================================

// A common interface for subgroup functions
// genuinely no idea how this maps to GLSL as of now; Vulkan probably has entirely different extensions
// We will just have to disable it host-side until a solution is found...
#if USE_SUBGROUP_SHUFFLING == 1

int clblast_get_sub_group_local_id()
{
	return get_sub_group_local_id();
}

realN clblast_sub_group_shuffle(realN reg, int src)
{
	return subgroupShuffle(reg, uint(src));
}
#endif

// Main body of the matrix-multiplication algorithm. It calls various (inlined) functions.
void XgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,
#if USE_BDA
	const __global realM* restrict agm, const __global realN* restrict bgm,
	__global realM* cgm,
#else
	int a_offset, int b_offset, int c_offset,
#endif
	const real alpha, const real beta)
{

	// Allocates workitem-private memory (registers)
	#if GEMMK == 0
		
		realM apm[MWI/VWM]; // MWI * 1
		
		realN bpm[NWI/VWN]; // 1 * NWI
	#elif GEMMK == 1
		#if USE_SUBGROUP_SHUFFLING == 1
			realN apm[KREG/VWN]; // KREG (subgroup shuffling in NWI dimension)
		#else
			
			realN apm[NWI*(KREG/VWN)]; // NWI * KREG
		#endif
		
		realM bpm[KREG*(MWI/VWM)]; // KREG * MWI
	#endif
	
	realM cpm[NWI*(MWI/VWM)]; // NWI * MWI

	#if GEMMK == 1
#if USE_BDA
		const __global real* restrict a_ptr = (const __global real* restrict) &agm[0];
		const __global real* restrict b_ptr = (const __global real* restrict) &bgm[0];
#else
		// use for scalar bgms
		int a_ptr_offset = a_offset*VWM;
		int b_ptr_offset = b_offset*VWN;
#endif
		const int tid_x = get_local_id(0) + MDIMC * GetGroupID0();
		const int tid_y = get_local_id(1) + NDIMC * GetGroupID1();
	#endif

	// Combined thread identifier (to disable caching)
	#if SA == 1 || SB == 1
		int tid = get_local_id(0) + MDIMC*get_local_id(1);
	#endif

	// Initializes the accumulation registers
	
	for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
		
		for (int _ni = 0; _ni < NWI; _ni += 1) {
			cpm[_ni * (MWI/VWM) + _mi] = InitAccRegisters();
		}
	}

	// Loops over all workgroup tiles
	for (int kwg = 0; kwg < kSizeK; kwg += KWG * KREG) {

		// Loads data: off-chip --> local (matrix A)
		#if SA == 1
			GlobalToLocalA(
#if USE_BDA
				agm,
#else
				a_offset,
#endif
				//alm,
				kSizeM, tid, kwg);
		#endif
		// Loads data: off-chip --> local (matrix B)
		#if SB == 1
			GlobalToLocalB(
#if USE_BDA
				bgm,
#else
				b_offset,
#endif
				//blm,
				kSizeN, tid, kwg);
		#endif
		#if SA == 1 || SB == 1
			barrier();
		#endif

		// Loops over all workitem tiles, unrolled by a factor KWI
		for (int pwi = 0; pwi < KWG * KREG; pwi += KWI * KREG) {
			
			for (int _pit = 0; _pit < KWI*KREG; _pit += KREG) {
				#if SA == 0 || SB == 0
					int idk = kwg + pwi + _pit;
				#endif
				#if SA == 1 || SB == 1
					int kg = pwi + _pit;
				#endif

				// Loads matrix A (kernel 0) or matrix B (kernel 1)
				
				for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
					// Loads data: local --> private (matrix A)
					#if GEMMK == 0 && SA == 1
						apm[_mi] = LocalToPrivateA(//alm,
							_mi, kg);
					// Loads data: off-chip --> private (matrix A)
					#elif GEMMK == 0 && SA == 0
						apm[_mi] = GlobalToPrivateA(
#if USE_BDA
							agm,
#else
							a_offset,
#endif
							_mi, kSizeM, idk, kwg);
					// Loads data: 2D global --> 2D private (matrix B)
					#elif GEMMK == 1
						
						for (int _ki = 0; _ki < KREG; _ki += 1) {
							bpm[_ki * (MWI/VWM) + _mi] = GlobalToPrivateB2D(
#if USE_BDA
								b_ptr,
#else
								b_ptr_offset,
#endif
								tid_x, _mi, kSizeN, idk, _ki);
						}
					#endif
				}

				// Loads matrix B (kernel 0) or matrix A (kernel 1)
				#if GEMMK == 0
					
					for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
						// Loads data: local --> private (matrix B)
						#if SB == 1
							bpm[_ni] = LocalToPrivateB(//blm,
								_ni, kg);
						// Loads data: off-chip --> private (matrix B)
						#else
							bpm[_ni] = GlobalToPrivateB(
#if USE_BDA
								bgm,
#else
								b_offset,
#endif
								_ni, kSizeN, idk);
						#endif
					}
				#elif GEMMK == 1
					// Loads data: 2D global --> 2D private (matrix A). Partly, shuffled later among subgroups
					#if USE_SUBGROUP_SHUFFLING == 1
						const int _ni = clblast_get_sub_group_local_id();
						
						for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
							apm[_ki] = GlobalToPrivateA2D(
#if USE_BDA
								a_ptr,
#else
								a_ptr_offset,
#endif
								tid_y, _ni, kSizeK, idk, _ki);
						}
					// Loads data: 2D global --> 2D private (matrix A)
					#else
						
						for (int _ni = 0; _ni < NWI; _ni += 1) {
							
							for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
								apm[_ni * (KREG/VWN) + _ki] = GlobalToPrivateA2D(
#if USE_BDA
									a_ptr,
#else
									a_ptr_offset,
#endif
									tid_y, _ni, kSizeK, idk, _ki);
							}
						}
					#endif
				#endif

				// Performs the accumulation (Cpm += Apm * Bpm)
				#if GEMMK == 0
					
					for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
						
						for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
							const realM aval = apm[_mi];
							#if VWN == 1
								cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni]);
							#else
								UNROLL(VWN)
								for (uint iv = 0; iv < VWN; iv += 1)
									cpm[(_ni*VWN + iv )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + iv )*(MWI/VWM) + _mi], aval, bpm[_ni].s[iv]);
							#endif
						}
					}
				#elif GEMMK == 1
					
					for (int _ni = 0; _ni < NWI; _ni += 1) {
						
						for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
							
							for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
								#if USE_SUBGROUP_SHUFFLING == 1
									const realN aval = clblast_sub_group_shuffle(apm[_ki], _ni);
								#else
									const realN aval = apm[_ni * (KREG/VWN) + _ki];
								#endif
								#if VWN == 1
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval);
								#else
									UNROLL(VWN)
									for (uint iv = 0; iv < VWN; iv += 1)
										cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + iv) * (MWI/VWM) + _mi], aval.s[iv]);
								#endif
							}
						}
					}
				#endif

			}
		}
		#if SA == 1 || SB == 1
			barrier();
		#endif
	}
	#if GLOBAL_MEM_FENCE == 1
		memoryBarrier(); barrier();
	#endif

	// Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
	#if GEMMK == 0
		const int cld = kSizeM;
	#elif GEMMK == 1
		const int cld = kSizeN;
	#endif
	
	for (int _ni = 0; _ni < NWI; _ni += 1) {
		
		for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
			StoreResults(
#if USE_BDA
				cgm,
#else
				c_offset,
#endif
				cpm[_ni * (MWI/VWM) + _mi], _mi, _ni, cld, alpha, beta);
		}
	}
}
#endif
// End of the C++11 raw string literal

// =================================================================================================

// The upper-triangular and lower-triangular kernels are only used in special cases

// Main entry point of the kernel. This is the lower-triangular version.
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = MDIMC, local_size_y = NDIMC, local_size_z = 1) in;
#endif

layout(push_constant, std430) uniform XgemmLower
{
	int kSizeN; int kSizeK;
	real_arg arg_alpha;
	real_arg arg_beta;
#if USE_BDA
	__global realM* restrict agm;
	__global realN* restrict bgm;
	__global realM* cgm;
#endif
};

void main()
{
	const real alpha = GetRealArg(arg_alpha);
	const real beta = GetRealArg(arg_beta);

	// Skip these threads if they do not contain threads contributing to the lower-triangle
	if (GetGroupID1()*NWG > (GetGroupID0() + 1)*MWG) {
		return;
	}

	// Computes the matrix-multiplication and stores the result in global memory
	XgemmBody(kSizeN, kSizeN, kSizeK,
#if USE_BDA
		agm, bgm, cgm,
#else
		0, 0, 0,
#endif
		alpha, beta
	);
}

// =================================================================================================


)"
// End of the C++11 raw string literal

// =================================================================================================
