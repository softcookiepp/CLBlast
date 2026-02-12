
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common functions and parameters specific for level 1 BLAS kernels.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

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

// End of the C++11 raw string literal
)"

// =================================================================================================
