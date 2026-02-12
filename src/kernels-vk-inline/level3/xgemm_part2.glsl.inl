
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

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

)"
// End of the C++11 raw string literal

// =================================================================================================
