
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

// Direct version of the GEMM kernel with [A, B] = [transposed, transposed]
// XgemmDirectTT
void main()
{
	XgemmDirect(args.kSizeM, args.kSizeN, args.kSizeK, args.arg_alpha, args.arg_beta,
#if USE_BDA
		agm,
#endif
		args.a_offset, args.a_ld,
#if USE_BDA
		bgm,
#endif
		args.b_offset, args.b_ld,
#if USE_BDA
		cgm,
#endif
		args.c_offset, args.c_ld,
		//alm, blm,
		1, 1, args.c_transpose, args.a_conjugate, args.b_conjugate);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
