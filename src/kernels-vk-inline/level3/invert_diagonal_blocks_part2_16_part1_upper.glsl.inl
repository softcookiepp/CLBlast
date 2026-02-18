

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 2, see part 1 of the invert kernel for a description
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains kernels to invert squared diagonal blocks of a matrix. These kernels are based
// on the TRSM implementation in the CUDA version of Magma version 2.2.0 and the poster "Triangular
// Linear System Solver for GPU with CUDA and OpenCL" by Peng Du, Stanimire Tomov, Piotr Luszczek,
// and Jack Dongarra.
//
// This is part 1 of 2, see part 2 for the remainder of the kernel code.
//
// =================================================================================================
//
//	Let A be an block_size*block_size lower triangular matrix, and B its inverse.
//	Then the block decomposition
//	
//			[ A11	 0	] * [ B11	 0	] = [ I 0 ]
//			[ A21	A22 ]	 [ B21	B22 ]	 [ 0 I ]
//	
//	yields
//	
//			A11*B11 = I						==>	B11 =	A11^{-1},
//			A22*B22 = I						==>	B22 =	A22^{-1},
//			A21*B11 + A22*B21 = 0	==>	B21 = -A22^{-1}*A21*B11 = -B22*A21*B11.
//	
//	The InvertDiagonalBlock kernel inverts A11 and A22.
//	The TripleMatMul routines multiply:
//	part 1:	B21 =	A21 * B11,
//	part 2:	B21 = -B22 * B21.
//	
//	At this level, inner block is current_size=16, with one 4 x 4 work-group per inner block. Each
//	submatrix Aij and Bij is current_size x current_size. The submatrix dimension is multiplied by 2
//	at each level, so the next level is current_size*2 = 32. A 'page' is the next bigger block,
//	here current_size*2=32,
//								 [ B11	 0	]
//	which contains [ B21	B22 ].
//	Outer blocks are block_size x block_size.
//	
//	A21 may have < current_size rows, but is guaranteed to have current_size cols since A22 is on
//	the right. This makes a single check easy to do.
//	
//	B is stored in workspace that is a full multiple of block_size x block_size; no checks needed.
//	
//	We split this into part1 & part2 to synchronize all blocks and make sure
//	that writes to B12 are observed by all blocks.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef INVERT_DIAGONAL_BLOCKS_PART1_TRIPLE_MATMUL_GLSL
#define INVERT_DIAGONAL_BLOCKS_PART1_TRIPLE_MATMUL_GLSL


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

// =================================================================================================
#if 1//defined(ROUTINE_INVERT)

// Parameters set by the tuner
// TODO: Make these actually tunable
#ifndef INTERNAL_BLOCK_SIZE
	#define INTERNAL_BLOCK_SIZE 16		 // Internal block size of the invert kernel
#endif
#ifndef LOCALPAD
	#define LOCALPAD 0								 // Padding in the x-dimension of the local memory to avoid bank conflicts
#endif
#ifndef LOCALX
	#define LOCALX (16 + LOCALPAD)		 // Local memory size in x-dimension of TripleMatMul kernels
#endif
#ifndef LOCALY
	#define LOCALY 16									// Local memory size in y-dimension of TripleMatMul kernels
#endif
#ifndef TMMWGSX
	#define TMMWGSX 4									// Work-group size in x-dimension of TripleMatMul kernels
#endif
#ifndef TMMWGSY
	#define TMMWGSY 4									// Work-group size in y-dimension of TripleMatMul kernels
#endif

// =================================================================================================

// Triple matrix-multiplication kernel: C = A * B
// oh god, here we go again with the huge macros
#define TripleMatMul(size, upper, part, blm, n, agm, agm_offset_init, bgm, bgm_offset_init, cgm, cgm_offset_init, lda, ldb, ldc, current_size, num_pages, block_size) \
{ \
	const int by	 = get_group_id(1) / num_pages; \
	const int page = get_group_id(1) % num_pages; \
	const int lidx = get_local_id(0); \
	const int lidy = get_local_id(1); \
	const int ibx	= get_group_id(0) * (get_local_size(0) * TMMWGSY); \
	const int iby	= by*16; \
	const int id	 = lidx + lidy*get_local_size(0); \
	const int row	= page*current_size*2 + current_size + ibx + id; \
	int col				= page*current_size*2 + current_size; \
	int agm_offset = agm_offset_init + ibx + id; \
	int bgm_offset = bgm_offset_init + lidx + (iby + lidy)*ldb; \
	int cgm_offset = cgm_offset_init + ibx + id + iby*ldc; \
	real cpm[16]; \
	for (int _j = 0; _j < 16; _j += 1) { \
		SetToZero(cpm[_j]); \
	} \
	for (int k = 0; k < current_size; k += 16) { \
		\
		for (int i = 0; i < 16; i += (size/4) ) { \
			for (int _j = 0; _j < 16; _j += TMMWGSY ) { \
				blm[(lidx + i) * LOCALX + (lidy + _j)] = bgm[k + i + _j*ldb + bgm_offset]; \
			} \
		} \
		barrier(); \
		if (upper) { \
			for (int _i = 0; _i < 16; _i += 1) { \
				if (part == 2 || col++ < n) { \
					for (int _j = 0; _j < 16; _j += 1) { \
						MultiplyAdd(cpm[_j], agm[(_i + k) * lda + agm_offset], blm[_i * LOCALX + _j]); \
					} \
				} \
			} \
		} \
		else { \
			if (row < n) { \
				for (int _i = 0; _i < 16; _i += 1) { \
					for (int _j = 0; _j < 16; _j += 1) { \
						MultiplyAdd(cpm[_j], agm[(_i + k) * lda + agm_offset], blm[_i * LOCALX + _j]); \
					} \
				} \
			} \
		} \
		barrier(); \
	} \
	for (int _i = 0; _i < 16; _i += 1) { \
		if (part == 2) { Negate(cpm[_i]); } \
		cgm[cgm_offset] = cpm[_i]; \
		cgm_offset += ldc; \
	} \
}

// =================================================================================================

// Triple matrix-multiplication kernel part 1: B12 = A12 * B22 (upper) or B21 = A21 * B11 (lower)
#define TripleMatMulPart1(size, upper, blm, n, src, a_offset, lda, dest, current_size, num_pages, block_size) \
{ \
	const int page = get_group_id(1) % num_pages; \
	const int pages_per_block = block_size / (current_size*2); \
	int dest_offset = (page / pages_per_block) * block_size * block_size + \
					(page % pages_per_block) * (current_size*2*block_size + current_size*2); \
	int agm_offset = 0; \
	int bgm_offset = 0; \
	int cgm_offset = 0; \
	\
	if (upper) { \
		agm_offset = a_offset + page*current_size*2*lda + page*current_size*2 + current_size*lda; \
		bgm_offset = dest_offset + current_size*block_size + current_size; \
		cgm_offset = dest_offset + current_size*block_size; \
	} \
	else { \
		agm_offset = a_offset + page*current_size*2*lda + page*current_size*2 + current_size; \
		bgm_offset = dest_offset; \
		cgm_offset = dest_offset + current_size; \
	} \
	const int ldb = block_size; \
	const int ldc = block_size; \
	TripleMatMul(size, upper, 1, blm, n, src, agm_offset, dest, bgm_offset, dest, cgm_offset, lda, ldb, ldc, current_size, num_pages, block_size); \
}

// Triple matrix-multiplication kernel part 2: B12 = -B11 * B12 (upper) or B21 = -B22 * B21 (lower)
#define TripleMatMulPart2(size, upper, blm, n, dest, current_size, num_pages, block_size) \
{ \
	const int page = get_group_id(1) % num_pages; \
	const int pages_per_block = block_size / (current_size*2); \
	int dest_offset = (page / pages_per_block) * block_size * block_size + \
					(page % pages_per_block) * (current_size*2*block_size + current_size*2); \
	int agm_offset = 0; \
	int bgm_offset = 0; \
	int cgm_offset = 0; \
	if (upper) { \
		agm_offset = dest_offset;\
		cgm_offset = dest_offset + current_size*block_size; \
		bgm_offset = cgm_offset; \
	} \
	else { \
		agm_offset = dest_offset + current_size*block_size + current_size; \
		cgm_offset = dest_offset + current_size; \
		bgm_offset = cgm_offset; \
	} \
	\
	const int lda = block_size; \
	const int ldb = block_size; \
	const int ldc = block_size; \
	TripleMatMul(size, upper, 2, blm, n, dest, agm_offset, dest, bgm_offset, dest, cgm_offset, lda, ldb, ldc, current_size, num_pages, block_size); \
}

#endif
// =================================================================================================
#endif
// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================

#if RELAX_WORKGROUP_SIZE == 0
	// local size appears to be variable for these kernels, so that is what we will do
	layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer src_buf { real src[]; };
	layout(binding = 1, std430) buffer dest_buf { real dest[]; };
#endif

layout(push_constant, std430) uniform TripleMatMulPart1
{
	int n;
#if USE_BDA
	__global const real* restrict src;
#endif
	int a_offset; int lda;
#if USE_BDA
	__global real* restrict dest;
#endif
	int current_size; int num_pages; int block_size;
} args;

shared real lm[LOCALY * LOCALX];

// B12 =	A12 * B22
// TripleMatMul16Part1Upper
void main()
{
	TripleMatMulPart1(16, true, lm, args.n, src, args.a_offset,
		args.lda, dest, args.current_size, args.num_pages, args.block_size);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
