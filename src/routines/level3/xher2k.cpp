
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher2k class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xher2k.hpp"

#include <cstddef>
#include <string>

#include "routines/level3/xherk.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
Xher2k::Xher2k(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name) : Xherk(dtype, queue, event, name) {}

// =================================================================================================

// The main routine
void Xher2k::DoHer2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose, const size_t n,
	const size_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset,
	const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld,
	const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld)
{
	
	
	// Runs the first matrix multiplication
	tart::Scalar complex_beta(std::complex<double>(beta.get<double>(), 0.0));
	const auto negated_ab_transpose = (ab_transpose != Transpose::kNo) ? Transpose::kNo : Transpose::kYes;
	HerkAB(layout, triangle, ab_transpose, negated_ab_transpose, n, k, alpha, a_buffer, a_offset, a_ld, b_buffer,
				 b_offset, b_ld, complex_beta, c_buffer, c_offset, c_ld, false);

	// Swaps the arguments for matrices A and B, sets 'beta' to 1, and conjugate alpha
	std::complex<double> alpha_value = alpha.getComplex<double>();
	tart::Scalar conjugate_alpha(std::complex<double>(alpha_value.real(), -alpha_value.imag()));
	auto complex_one = tart::Scalar::one();
	HerkAB(layout, triangle, ab_transpose, negated_ab_transpose, n, k, conjugate_alpha, b_buffer, b_offset, b_ld,
				 a_buffer, a_offset, a_ld, complex_one, c_buffer, c_offset, c_ld, true);
				 
	
}

// =================================================================================================
}	// namespace clblast
