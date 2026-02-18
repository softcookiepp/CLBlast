
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xhad kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=incz=1) and no offsets (offx=offy=offz=0). Another version is more general,
// but doesn't support vector data-types. Based on the XAXPY kernels.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

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
		#error "not implemented!"
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
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE) buffer real_ptr_t { real s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*2) buffer real2_ptr_t { real2 s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*4) buffer real4_ptr_t { real4 s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*8) buffer real8_ptr_t { real8 s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*16) buffer real16_ptr_t { real16 s[]; };
	
	// this will not work for addresses represented as uvec2; those still need to be implemented
	#define INDEX_AS_ALIGNED(buffer_t, ptr, index, alignment) buffer_t(addPtrOffset(addr_t(ptr), index*alignment))
	real_ptr_t indexGMimpl(real_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real_ptr_t, ptr, index, DTYPE_SIZE); }
	real2_ptr_t indexGMimpl(real2_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real2_ptr_t, ptr, index, DTYPE_SIZE*2); }
	real4_ptr_t indexGMimpl(real4_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real4_ptr_t, ptr, index, DTYPE_SIZE*4); }
	real8_ptr_t indexGMimpl(real8_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real8_ptr_t, ptr, index, DTYPE_SIZE*8); }
	real16_ptr_t indexGMimpl(real16_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real16_ptr_t, ptr, index, DTYPE_SIZE*16); }
	#define indexGM(ptr, index) indexGMimpl(ptr, index).s[0]
#else
	#define indexGM(ptr, index) ptr[index]
#endif

// Single-element version of a complex number
#if PRECISION == 3232
	#define singlereal float 
#elif PRECISION == 6464
	#define singlereal double 
#else
	#define singlereal real 
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
// This file contains the common functions and parameters specific for level 1 BLAS kernels.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef LEVEL1_GLSL
#define LEVEL1_GLSL


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
		#error "not implemented!"
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
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE) buffer real_ptr_t { real s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*2) buffer real2_ptr_t { real2 s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*4) buffer real4_ptr_t { real4 s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*8) buffer real8_ptr_t { real8 s[]; };
	layout(buffer_reference, buffer_reference_align = DTYPE_SIZE*16) buffer real16_ptr_t { real16 s[]; };
	
	// this will not work for addresses represented as uvec2; those still need to be implemented
	#define INDEX_AS_ALIGNED(buffer_t, ptr, index, alignment) buffer_t(addPtrOffset(addr_t(ptr), index*alignment))
	real_ptr_t indexGMimpl(real_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real_ptr_t, ptr, index, DTYPE_SIZE); }
	real2_ptr_t indexGMimpl(real2_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real2_ptr_t, ptr, index, DTYPE_SIZE*2); }
	real4_ptr_t indexGMimpl(real4_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real4_ptr_t, ptr, index, DTYPE_SIZE*4); }
	real8_ptr_t indexGMimpl(real8_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real8_ptr_t, ptr, index, DTYPE_SIZE*8); }
	real16_ptr_t indexGMimpl(real16_ptr_t ptr, uint index) { return INDEX_AS_ALIGNED(real16_ptr_t, ptr, index, DTYPE_SIZE*16); }
	#define indexGM(ptr, index) indexGMimpl(ptr, index).s[0]
#else
	#define indexGM(ptr, index) ptr[index]
#endif

// Single-element version of a complex number
#if PRECISION == 3232
	#define singlereal float 
#elif PRECISION == 6464
	#define singlereal double 
#else
	#define singlereal real 
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

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef WGS
	#define WGS 64		 // The local work-group size
#endif
#ifndef WPT
	#define WPT 1			// The amount of work-per-thread
#endif
#ifndef VW
	#define VW 1			 // Vector width of vectors X and Y
#endif

// =================================================================================================

// Data-widths
#if VW == 1
	#define realV real
#elif VW == 2
	#define realV real2
#elif VW == 4
	#define realV real4
#elif VW == 8
	#define realV real8
#elif VW == 16
	#define realV real16
#endif

// =================================================================================================

// The vectorized multiply function
realV MultiplyVector(realV cvec, const real aval, const realV bvec) {
	#if VW == 1
		Multiply(cvec, aval, bvec);
	#else
		vsMultiply(cvec, aval, bvec, VW);
	#endif
	return cvec;
}

// The vectorized multiply-add function
realV MultiplyAddVector(realV cvec, const real aval, const realV bvec) {
	#if VW == 1
		MultiplyAdd(cvec, aval, bvec);
	#else
		vsMultiplyAdd(cvec, aval, bvec, VW);
	#endif
	return cvec;
}

// =================================================================================================
#endif
// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================

// A vector-vector multiply function. See also level1.opencl for a vector-scalar version
realV MultiplyVectorVector(realV cvec, const realV aval, const realV bvec) {
	#if VW == 1
		Multiply(cvec, aval, bvec);
	#else
		vMultiply(cvec, aval, bvec, VW);
	#endif
	return cvec;
}

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { realV xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { realV ygm[]; };
	layout(binding = 2, std430) buffer zgm_buf { realV zgm[]; };
#endif

layout(push_constant) uniform XhadFaster
{
	int n; real_arg arg_alpha; real_arg arg_beta;
#if USE_BDA
	__global real* restrict xgm;
	__global real* restrict ygm;
	__global real* zgm;
#endif
};

// Faster version of the kernel without offsets and strided accesses but with if-statement. Also
// assumes that 'n' is dividable by 'VW' and 'WPT'.

void main()
{
	if (!(n % VW == 0 && n % WPT == 0)) return;
	
	const real alpha = GetRealArg(arg_alpha);
	const real beta = GetRealArg(arg_beta);

	const int num_desired_threads = n / (VW * WPT);

	if (get_global_id(0) < num_desired_threads) {
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id = _w * num_desired_threads + get_global_id(0);
			realV xvalue = xgm[id];
			realV yvalue = ygm[id];
			realV zvalue = zgm[id];
			realV result;
			realV alpha_times_x;
			alpha_times_x = MultiplyVector(alpha_times_x, alpha, xvalue);
			result = MultiplyVectorVector(result, alpha_times_x, yvalue);
			zgm[id] = MultiplyAddVector(result, beta, zvalue);
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
