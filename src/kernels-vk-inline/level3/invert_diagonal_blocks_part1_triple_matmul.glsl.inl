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
R"(

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

// End of the C++11 raw string literal
)"

// =================================================================================================
