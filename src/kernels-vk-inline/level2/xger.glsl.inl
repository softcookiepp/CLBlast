

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xger kernels for rank-1 matrix update.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef COMMON_GLSL
#define COMMON_GLSL
// =================================================================================================

#define USE_BDA 0

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

// Enable support for half-precision
#if PRECISION == 16
	#extension GL_EXT_shader_16bit_storage : require
	#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

// Enable support for double-precision
#if PRECISION == 64 || PRECISION == 6464
	#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#endif


// Half-precision
#if PRECISION == 16
	struct vec2_t { f16vec2 s; };
	struct vec4_t { f16vec4 s; };
	struct vec8_t { float16_t s[8]; };
	struct vec16_t { float16_t s[16]; };
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
	#if 0 //USE_CL_MAD == 1
		#define MultiplyAdd(c,a,b) c = mad(a, b, c)
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
	#define INDEX(buf, idx) buf[idx]
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
// This file contains common functions for matrix update kernels (Xger, Xher).
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef LEVEL2_GLSL
#define LEVEL2_GLSL
// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

#ifndef WGS1
	#define WGS1 8		// The local work-group size in first dimension
#endif
#ifndef WGS2
	#define WGS2 8		// The local work-group size in second dimension
#endif
#ifndef WPT
	#define WPT 1		 // The amount of work-per-thread in both dimensions
#endif

// =================================================================================================

// Returns an element from a vector
real LoadVectorImpl(real result, const int id, const int max, const int offset, const int inc,
														const bool do_conjugate) {
	if (id < max) {
		//real result = gm[id*inc + offset];
		if (do_conjugate) {
			#if defined(ROUTINE_GERC) || defined(ROUTINE_HER) || defined(ROUTINE_HPR) || defined(ROUTINE_HER2) || defined(ROUTINE_HPR2)
				COMPLEX_CONJUGATE(result);
			#endif
		}
		return result;
	}
	else {
		real default_result;
		SetToZero(default_result);
		return default_result;
	}
}

#define LoadVector(final_result, id, max, gm, offset, inc, do_conjugate) \
{ \
	real result = gm[id*inc + offset]; \
	final_result = LoadVectorImpl(result, id, max, offset, inc, do_conjugate); \
}

int GetMatrixUpdateIndex(const int id1, const int id2, const int a_offset, const int a_ld, const bool is_upper)
{
	#if defined(ROUTINE_SPR) || defined(ROUTINE_HPR)
		int a_index;
		if (is_upper) {
			a_index = (id1 <= id2) ? ((id2+1)*id2)/2 + id1 : ((id1+1)*id1)/2 + id2;
		}
		else {
			a_index = (id1 >= id2) ? ((2*a_ld-(id2+1))*id2)/2 + id1 : ((2*a_ld-(id1+1))*id1)/2 + id2;
		}
		a_index += a_offset;
		return a_index;
	#else
		return id2*a_ld + id1 + a_offset;
	#endif
}

// Performs the rank-1 matrix update
real MatrixUpdateImpl(const int id1, const int id2, const int max1, const int max2,
															real avalue, const int a_offset, const int a_ld,
															const real alpha, const real xvalue, const real yvalue,
															const bool is_upper)
{
	// Computes result = alpha * x[i] * y[j] + a[i][j]
	#if PRECISION == 3232 || PRECISION == 6464
		real ax;
		ax.x = MulReal(alpha, xvalue);
		ax.y = MulImag(alpha, xvalue);
		real result;
		result.x = MulReal(ax, yvalue) + avalue.x;
		result.y = MulImag(ax, yvalue) + avalue.y;
	#else
		real result = alpha * xvalue * yvalue + avalue;
	#endif

	// For hermetian matrices
	#if defined(ROUTINE_HER) || defined(ROUTINE_HPR)
		if (id1 == id2) { result.y = ZERO; }
	#endif
	
	// Stores the final result
	//agm[a_index] = result;
	return result;
}

#define MatrixUpdate(id1, id2, max1, max2, agm, a_offset, a_ld, alpha, xvalue, yvalue, is_upper) \
{ \
	if (id1 < max1 && id2 < max2) \
	{ \
		const int a_index = GetMatrixUpdateIndex(id1, id2, a_offset, a_ld, is_upper); \
		agm[a_index] = MatrixUpdateImpl(id1, id2, max1, max2, agm[a_index], a_offset, a_ld, alpha, xvalue, yvalue, bool(is_upper)); \
	} \
}

int GetMatrixUpdate2Index(const int id1, const int id2, const int a_offset, const int a_ld, const bool is_upper)
{
	#if defined(ROUTINE_SPR2) || defined(ROUTINE_HPR2)
		int a_index;
		if (is_upper) {
			a_index = (id1 <= id2) ? ((id2+1)*id2)/2 + id1 : ((id1+1)*id1)/2 + id2;
		}
		else {
			a_index = (id1 >= id2) ? ((2*a_ld-(id2+1))*id2)/2 + id1 : ((2*a_ld-(id1+1))*id1)/2 + id2;
		}
		a_index += a_offset;
		return a_index;
	#else
		return id2*a_ld + id1 + a_offset;
	#endif
}

// main body of matrix update 2
real MatrixUpdate2Impl(const int id1, const int id2, const int max1, const int max2,
															 const real avalue, const int a_offset, const int a_ld,
															 const real alpha1, const real xvalue, const real yvalue,
															 const real alpha2, const real xtvalue, const real ytvalue,
															 const bool is_upper)
{
	// Computes result = alpha * x[i] * y[j] + alpha * x[j] * y[i] + a[i][j]
	#if PRECISION == 3232 || PRECISION == 6464
		real ax;
		ax.x = MulReal(alpha2, xvalue);
		ax.y = MulImag(alpha2, xvalue);
		real atx;
		atx.x = MulReal(alpha1, xtvalue);
		atx.y = MulImag(alpha1, xtvalue);
		real result;
		result.x = MulReal(ax, yvalue) + MulReal(atx, ytvalue) + avalue.x;
		result.y = MulImag(ax, yvalue) + MulImag(atx, ytvalue) + avalue.y;
	#else
		real result = alpha1 * xvalue * yvalue + alpha2 * xtvalue * ytvalue + avalue;
	#endif

	// For hermetian matrices
	#if defined(ROUTINE_HER2) || defined(ROUTINE_HPR2)
		if (id1 == id2 && (alpha1.x > 0.0 || alpha1.y > 0.0 || alpha2.x > 0 || alpha2.y > 0)) { result.y = ZERO; }
	#endif

	// Stores the final result
	return result;
}

// Performs the rank-2 matrix update
// needs to be a macro because GLSL is stupid about passing buffers
#define MatrixUpdate2(id1, id2, max1, max2, agm, a_offset, a_ld, \
	alpha1, xvalue, yvalue, alpha2, xtvalue, ytvalue, is_upper) \
{ \
	if (id1 < max1 && id2 < max2) \
	{ \
		const int a_index = GetMatrixUpdate2Index(id1, id2, a_offset, a_ld, is_upper); \
		const real avalue = agm[a_index]; \
		const real result = MatrixUpdate2Impl(id1, id2, max1, max2, avalue, a_offset, a_ld, \
			alpha1, xvalue, yvalue, alpha2, xtvalue, ytvalue, bool(is_upper)); \
		agm[a_index] = result; \
	} \
}

// =================================================================================================
#endif
// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================

// Regular version of the rank-1 matrix update kernel (GER, GERU, GERC)
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS1, local_size_y = WGS2, local_size_z = 1) in;
	//__kernel __attribute__((reqd_work_group_size(WGS1, WGS2, 1)))
#endif

#if USE_BDA
#else
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { real ygm[]; };
	layout(binding = 2, std430) buffer agm_buf { real agm[]; };
#endif

layout(push_constant) uniform Xger
{
	int max1;
	int max2;
	real_arg arg_alpha;
#if USE_BDA
	__global real* restrict xgm;
#endif
	int x_offset;
	int x_inc;
#if USE_BDA
	__global real* ygm;
#endif
	int y_offset;
	int y_inc;
#if USE_BDA
	__global real* restrict agm;
#endif
	int a_offset;
	int a_ld;
	int is_rowmajor;
} args;

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);

	// Register storage for X and Y
	//#pragma promote_to_registers
	real xvalues[WPT];
	//#pragma promote_to_registers
	real yvalues[WPT];

	// Row-major version
	if (bool(args.is_rowmajor)) {

		// Loads the X-vector
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id2 = _w*get_global_size(1) + get_global_id(1);
#if 1
			LoadVector(xvalues[_w], id2, args.max2, xgm, args.x_offset, args.x_inc, false);
#else
			xvalues[_w] = LoadVector(id2, args.max2, xgm, args.x_offset, args.x_inc, false);
#endif
		}

		// Loads the Y-vector
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id1 = _w*get_global_size(0) + get_global_id(0);
#if 1
			LoadVector(yvalues[_w], id1, args.max1, ygm, args.y_offset, args.y_inc, true);
#else
			yvalues[_w] = LoadVector(id1, args.max1, ygm, args.y_offset, args.y_inc, true);
#endif
		}

		// Loops over the work per thread twice
		//#pragma unroll
		for (int _w1 = 0; _w1 < WPT; _w1 += 1) {
			//#pragma unroll
			for (int _w2 = 0; _w2 < WPT; _w2 += 1) {

				// Global thread IDs
				const int id1 = _w1*get_global_size(0) + get_global_id(0);
				const int id2 = _w2*get_global_size(1) + get_global_id(1);

				// Loads A, performs the operation, and stores the result into A
				MatrixUpdate(id1, id2, args.max1, args.max2, agm, args.a_offset, args.a_ld,
										 alpha, xvalues[_w2], yvalues[_w1], false);
			}
		}
	}

	// Col-major version
	else {

		// Loads the X-vector
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id1 = _w*get_global_size(0) + get_global_id(0);
#if 1
			LoadVector(xvalues[_w], id1, args.max1, xgm, args.x_offset, args.x_inc, false);
#else
			xvalues[_w] = LoadVector(id1, args.max1, xgm, args.x_offset, args.x_inc, false);
#endif
		}

		// Loads the Y-vector
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id2 = _w*get_global_size(1) + get_global_id(1);
#if 1
			LoadVector(yvalues[_w], id2, args.max2, ygm, args.y_offset, args.y_inc, true);
#else
			yvalues[_w] = LoadVector(id2, args.max2, ygm, args.y_offset, args.y_inc, true);
#endif
		}

		// Loops over the work per thread twice
		//#pragma unroll
		for (int _w1 = 0; _w1 < WPT; _w1 += 1) {
			//#pragma unroll
			for (int _w2 = 0; _w2 < WPT; _w2 += 1) {

				// Global thread IDs
				const int id1 = _w1*get_global_size(0) + get_global_id(0);
				const int id2 = _w2*get_global_size(1) + get_global_id(1);

				// Loads A, performs the operation, and stores the result into A
				MatrixUpdate(id1, id2, args.max1, args.max2, agm, args.a_offset, args.a_ld,
										 alpha, xvalues[_w1], yvalues[_w2], false);
			}
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
