
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(

// The vectorised multiply-add function
INLINE_FUNC realM MultiplyAddVector(realM cvec, const realM avec, const real bval) {
	#if USE_VECTOR_MAD == 1
		cvec += avec * bval;
	#else
		#if VWM == 1
			MultiplyAdd(cvec, avec, bval);
		#elif VWM == 2
			MultiplyAdd(cvec.x , avec.x,	bval);
			MultiplyAdd(cvec.y , avec.y,	bval);
		#elif VWM == 4
			MultiplyAdd(cvec.x , avec.x,	bval);
			MultiplyAdd(cvec.y , avec.y,	bval);
			MultiplyAdd(cvec.z , avec.z,	bval);
			MultiplyAdd(cvec.w , avec.w,	bval);
		#elif VWM == 8
			MultiplyAdd(cvec[0], avec[0], bval);
			MultiplyAdd(cvec[1], avec[1], bval);
		#elif VWM == 16
			MultiplyAdd(cvec[0], avec[0], bval);
			MultiplyAdd(cvec[1], avec[1], bval);
			MultiplyAdd(cvec[2], avec[2], bval);
			MultiplyAdd(cvec[3], avec[3], bval);
		#endif
	#endif
	return cvec;
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
void StoreResults(
#if USE_BDA
		__global realM* cgm,
#endif
		realM c_value, const int _mi, const int _ni,
		const int kSizeM, const real alpha, const real beta)
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
	int idm = mg + GetGroupID0() * (MWG/VWM);
	int idn = ng + GetGroupID1() * NWG;
	int index = idn*(kSizeM/VWM) + idm;

	realM result;
	realM xval = c_value;

	// The final multiplication with alpha (in case beta == 0)
	if (IsZero(beta)) {
		#if VWM == 1
			Multiply(result, alpha, xval);
		#elif VWM == 2
			Multiply(result.x, alpha, xval.x);
			Multiply(result.y, alpha, xval.y);
		#elif VWM == 4
			Multiply(result.x, alpha, xval.x);
			Multiply(result.y, alpha, xval.y);
			Multiply(result.z, alpha, xval.z);
			Multiply(result.w, alpha, xval.w);
		#elif VWM == 8
			Multiply(result[0], alpha, xval[0]);
			Multiply(result[1], alpha, xval[1]);
		#elif VWM == 16
			Multiply(result[0], alpha, xval[0]);
			Multiply(result[1], alpha, xval[1]);
			Multiply(result[2], alpha, xval[2]);
			Multiply(result[3], alpha, xval[3]);
		#endif
	}

	// The final multiplication with alpha and the addition with beta*C
	else {
		realM yval = cgm[index];
		#if VWM == 1
			AXPBY(result, alpha, xval, beta, yval);
		#elif VWM == 2
			AXPBY(result.x, alpha, xval.x, beta, yval.x);
			AXPBY(result.y, alpha, xval.y, beta, yval.y);
		#elif VWM == 4
			AXPBY(result.x, alpha, xval.x, beta, yval.x);
			AXPBY(result.y, alpha, xval.y, beta, yval.y);
			AXPBY(result.z, alpha, xval.z, beta, yval.z);
			AXPBY(result.w, alpha, xval.w, beta, yval.w);
		#elif VWM == 8
			AXPBY(result[0], alpha, xval[0], beta, yval[0]);
			AXPBY(result[1], alpha, xval[1], beta, yval[1]);
		#elif VWM == 16
			AXPBY(result[0], alpha, xval[0], beta, yval[0]);
			AXPBY(result[1], alpha, xval[1], beta, yval[1]);
			AXPBY(result[2], alpha, xval[2], beta, yval[2]);
			AXPBY(result[3], alpha, xval[3], beta, yval[3]);
		#endif
	}
	cgm[index] = result;
}

//)"
// End of the C++11 raw string literal

// =================================================================================================
