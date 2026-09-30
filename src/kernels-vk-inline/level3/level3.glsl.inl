
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common functions and parameters specific for level 3 BLAS kernels.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
#ifndef LEVEL3_GLSL
#define LEVEL3_GLSL
// =================================================================================================

// Eventually, a lot of stuff will be replaced with specialization constants.
// But for now, the boilerplate for that has yet to be written.
#ifndef ROUTINE_SYRK
	#define ROUTINE_SYRK 0
#endif
#ifndef ROUTINE_HERK
	#define ROUTINE_HERK 0
#endif
#ifndef ROUTINE_SYR2K
	#define ROUTINE_SYR2K 0
#endif
#ifndef ROUTINE_HER2K
	#define ROUTINE_HER2K 0
#endif


// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

#ifndef LEVEL3_USE_SPEC
	#define LEVEL3_USE_SPEC 0
#endif

#if LEVEL3_USE_SPEC == 1
		// For the 'fast' copy kernel
	#ifdef COPY_DIMX
		#undef COPY_DIMX 			// Local workgroup size in the first dimension (x)
	#endif
	layout(constant_id = 0) const int COPY_DIMX = 8;
	
	#ifdef COPY_DIMY
		#undef COPY_DIMY 			// Local workgroup size in the second dimension (y)
	#endif
	layout(constant_id = 1) const int COPY_DIMY = 8;
	
	#ifdef COPY_WPT
		#undef COPY_WPT 			 // Work per thread in the first dimension (x)
	#endif
	layout(constant_id = 2) const int COPY_WPT = 1;
	
	#ifndef COPY_VW // un-specable
		#define COPY_VW 1				// Vector width in the second dimension (y)
	#endif
	

	// For the padding/copy kernels and the conversion kernels
	#ifdef PAD_DIMX
		#undef PAD_DIMX 			// Local workgroup size in the first dimension (x)
	#endif
	layout(constant_id = 3) const int PAD_DIMX = 8;
	
	#ifdef PAD_DIMY
		#undef PAD_DIMY 			// Local workgroup size in the second dimension (y)
	#endif
	layout(constant_id = 4) const int PAD_DIMY = 8;
	
	#ifdef PAD_WPTX
		#undef PAD_WPTX 			// Work per thread in the first dimension (x)
	#endif
	layout(constant_id = 5) const int PAD_WPTX = 1;
	
	#ifdef PAD_WPTY
		#undef PAD_WPTY 			// Work per thread in the second dimension (y)
	#endif
	layout(constant_id = 6) const int PAD_WPTY = 1;

	// For the 'fast' transpose kernel
	#ifdef TRA_DIM
		#undef TRA_DIM 			 // Number of local threads in the two dimensions (x,y)
	#endif
	layout(constant_id = 7) const int TRA_DIM = 8;
	
	#ifndef TRA_WPT // un-specable
		#define TRA_WPT 1			 // Work per thread in one dimension and vector-width in the other
	#endif
	
	#ifdef TRA_PAD
		#undef TRA_PAD 			 // Padding of the local memory to avoid bank-conflicts
	#endif
	layout(constant_id = 8) const int TRA_PAD = 0;
	
	#ifdef TRA_SHUFFLE
		#undef TRA_SHUFFLE 	 // Shuffling of the global indices to avoid global memory bank-conflicts
	#endif
	layout(constant_id = 9) const int TRA_SHUFFLE = 0;


	// For the padding/transpose kernels
	#ifdef PADTRA_TILE
		#undef PADTRA_TILE 	 // Number of local threads in the two dimensions (x,y)
	#endif
	layout(constant_id = 10) const int PADTRA_TILE = 8;
	
	#ifdef PADTRA_WPT
		#undef PADTRA_WPT 		// Amount of work per thread
	#endif
	layout(constant_id = 11) const int PADTRA_WPT = 1;
	
	#ifdef PADTRA_PAD
		#undef PADTRA_PAD 		// Padding of the local memory to avoid bank-conflicts
	#endif
	layout(constant_id = 12) const int PADTRA_PAD = 0;
	
#else
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
	#ifndef TRA_WPT // un-specable
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
#endif

// =================================================================================================
#endif
// End of the C++11 raw string literal
)"

// =================================================================================================
