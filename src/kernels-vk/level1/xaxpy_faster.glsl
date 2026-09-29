#version 450

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xaxpy kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=1) and no offsets (offx=offy=0). Another version is more general, but doesn't
// support vector data-types. The general version has a batched implementation as well.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"
// Enable this flag in order to pass vector width via specialization constant.
// This would eliminate the need for compilation of a different SPIR-V file for each vector width.
// It still needs to be performance-profiled before being introduced permanently, as it requires substantially more code.
#define USE_SPEC_FOR_VW 0
#include "level1.glsl"
// =================================================================================================

// Faster version of the kernel without offsets and strided accesses but with if-statement. Also
// assumes that 'n' is dividable by 'VW' and 'WPT'.
layout(local_size_x_id = 0) in;

#if USE_BDA == 0
	#if USE_SPEC_FOR_VW
		layout(binding = 0, std430) readonly buffer xgm_buf1 { real1 xgm1[]; };
		layout(binding = 0, std430) readonly buffer xgm_buf2 { real2 xgm2[]; };
		layout(binding = 0, std430) readonly buffer xgm_buf4 { real4 xgm4[]; };
		layout(binding = 0, std430) readonly buffer xgm_buf8 { real8 xgm8[]; };
		layout(binding = 0, std430) readonly buffer xgm_buf16 { real16 xgm16[]; };
		layout(binding = 1, std430) buffer ygm_buf1 { real1 ygm1[]; };
		layout(binding = 1, std430) buffer ygm_buf2 { real2 ygm2[]; };
		layout(binding = 1, std430) buffer ygm_buf4 { real4 ygm4[]; };
		layout(binding = 1, std430) buffer ygm_buf8 { real8 ygm8[]; };
		layout(binding = 1, std430) buffer ygm_buf16 { real16 ygm16[]; };
		
		#define loadXgm(x, id) \
		{ \
			[[flatten]] \
			if (VW == 1) { real1 xp = xgm1[id]; copyArbitraryVector(x, xp, VW); } \
			else if (VW == 2) { real2 xp = xgm2[id]; copyArbitraryVector(x, xp, VW); } \
			else if (VW == 4) { real4 xp = xgm4[id]; copyArbitraryVector(x, xp, VW); } \
			else if (VW == 8) { real8 xp = xgm8[id]; copyArbitraryVector(x, xp, VW); } \
			else if (VW == 16) { real16 xp = xgm16[id]; copyArbitraryVector(x, xp, VW); } \
		}
		
		#define loadYgm(y, id) \
		{ \
			[[flatten]] \
			if (VW == 1) { real1 yp = ygm1[id]; copyArbitraryVector(y, yp, VW); } \
			else if (VW == 2) { real2 yp = ygm2[id]; copyArbitraryVector(y, yp, VW); } \
			else if (VW == 4) { real4 yp = ygm4[id]; copyArbitraryVector(y, yp, VW); } \
			else if (VW == 8) { real8 yp = ygm8[id]; copyArbitraryVector(y, yp, VW); } \
			else if (VW == 16) { real16 yp = ygm16[id]; copyArbitraryVector(y, yp, VW); } \
		}
		
		void storeYgm(int id, realV y)
		{
			[[flatten]]
			if (VW == 1)
			{
				real1 yp;
				copyArbitraryVector(yp, y, VW);
				ygm1[id] = yp;
			}
			else if (VW == 2)
			{
				real2 yp;
				copyArbitraryVector(yp, y, VW);
				ygm2[id] = yp;
			}
			else if (VW == 4)
			{
				real4 yp;
				copyArbitraryVector(yp, y, VW);
				ygm4[id] = yp;
			}
			else if (VW == 8)
			{
				real8 yp = ygm8[id];
				copyArbitraryVector(yp, y, VW);
				ygm8[id] = yp;
			}
			else if (VW == 16)
			{
				real16 yp = ygm16[id];
				copyArbitraryVector(yp, y, VW);
				ygm16[id] = yp;
			}
		}
	#else
		layout(binding = 0, std430) readonly buffer xgm_buf { realV xgm[]; };
		layout(binding = 1, std430) buffer ygm_buf { realV ygm[]; };
	#endif
#endif

layout(push_constant) uniform XaxpyFaster
{
	int n;
	real_arg arg_alpha;
#if USE_BDA
	realV_ptr_t xgm;
	realV_ptr_t ygm;
#endif
};

void main()
{
	const real alpha = GetRealArg(arg_alpha);

	const int num_usefull_threads = n / (VW * WPT);
	if (get_global_id(0) < num_usefull_threads) {
		UNROLL(WPT)
		for (int _w = 0; _w < WPT; _w += 1)
		{
			const int id = _w*num_usefull_threads + get_global_id(0);
			#if USE_SPEC_FOR_VW
				realV xvalue;
				loadXgm(xvalue, id);
				realV yvalue;
				loadYgm(yvalue, id);
				vsMultiplyAdd(yvalue, alpha, xvalue, VW);
				storeYgm(id, yvalue);
			#else
				realV xvalue = indexGM(xgm, id);
				realV yvalue = indexGM(ygm, id);
				indexGM(ygm, id) = MultiplyAddVector(yvalue, alpha, xvalue);
			#endif
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
