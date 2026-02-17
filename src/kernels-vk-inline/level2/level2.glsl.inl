
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains common functions for matrix update kernels (Xger, Xher).
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
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
)"

// =================================================================================================
