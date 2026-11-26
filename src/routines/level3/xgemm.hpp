
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xgemm routine (general matrix-matrix multiplication).
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMM_H_
#define CLBLAST_ROUTINES_XGEMM_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"
#include "tart/dtypes.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Defines the assumptions of the GEMM kernels
inline bool a_want_rotated_(const size_t gemm_kernel_id) { return gemm_kernel_id == 1; }
inline bool b_want_rotated_(const size_t) { return true; }
inline bool c_want_rotated_(const size_t gemm_kernel_id) { return gemm_kernel_id == 1; }

// Computes the size of the temporary GEMM buffer based on user-arguments
size_t GetTempSize(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                   const uint32_t m, const uint32_t n, const uint32_t k, const size_t a_offset, const size_t a_ld,
                   const size_t b_offset, const size_t b_ld, const size_t c_offset, const size_t c_ld,
                   const size_t mwg, const size_t nwg, const size_t kwg, const size_t gemm_kernel_id);

// Selects which version of GEMM to run
bool UseDirectKernel(const uint32_t m, const uint32_t n, const uint32_t k, const size_t min_indirect_size);

// Process the user-arguments, computes secondary parameters
void ProcessArguments(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                      const uint32_t m, const uint32_t n, const uint32_t k, size_t& a_one, size_t& a_two,
                      size_t& b_one, size_t& b_two, size_t& c_one, size_t& c_two, bool& a_do_transpose,
                      bool& b_do_transpose, bool& c_do_transpose, bool& a_conjugate, bool& b_conjugate,
                      const size_t gemm_kernel_id);

// Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
size_t ComputeTempSize(const bool a_no_temp, const bool b_no_temp, const bool c_no_temp, const size_t a_size,
                       const size_t b_size, const size_t c_size, size_t& b_temp_offset, size_t& c_temp_offset);

// Determines whether or not temporary matrices are needed
inline bool NoTempBuffer(const size_t one, const size_t one_i, const size_t two, const size_t two_i, const uint32_t ld,
                         const uint32_t offset, const bool do_transpose, const bool conjugate) {
    return one == one_i && two == two_i && ld == one && offset == 0 && !do_transpose && !conjugate;
}

// Computes the first and second "internal" (ceiled) dimensions of the 3 matrices
void CalculateInternalDimensions(const uint32_t m, const uint32_t n, const uint32_t k, const size_t mwg,
                                 const size_t nwg, const size_t kwg, size_t& a_one_i, size_t& a_two_i,
                                 size_t& b_one_i, size_t& b_two_i, size_t& c_one_i, size_t& c_two_i,
                                 const size_t gemm_kernel_id);

// Functional implementation of Xgemm (general matrix-matrix multiplication)
void DoGemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose, const uint32_t m,
            const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset,
            const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld,
            const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
            const tart::buffer_ptr& temp_buffer, const bool temp_buffer_provided, const Databases& db,
            tart::device_ptr device, const tart::DType& dtype);

// Indirect version of GEMM (with pre and post-processing kernels)
void GemmIndirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                  const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
                  const size_t b_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
                  const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                  const bool a_conjugate, const bool b_conjugate, const size_t a_one, const size_t a_two,
                  const size_t b_one, const size_t b_two, const size_t c_one, const size_t c_two,
                  const tart::buffer_ptr& temp_buffer, const bool temp_buffer_provided, const Databases& db,
                  tart::device_ptr device, const tart::DType& dtype);

// Direct version of GEMM (no pre and post-processing kernels)
void GemmDirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
                const size_t b_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
                const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                const bool a_conjugate, const bool b_conjugate, const Databases& db,
                tart::device_ptr device, const tart::DType& dtype);

// Minimal base class for Xgemm-derived routines
class Xgemm {
 public:
	// Constructor
	Xgemm(const tart::DType& dtype, Queue& queue, const std::string& name = "GEMM");

	// Templated-precision implementation of the routine
	void DoGemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose, const uint32_t m,
							const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset,
							const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld,
							const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
							const tart::buffer_ptr& temp_buffer = tart::buffer_ptr(0), const bool temp_buffer_provided = false,
							const Databases& db = Databases({"Gemm"}),
							tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);

	// Indirect version of GEMM (with pre and post-processing kernels)
	void GemmIndirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
										const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
										const size_t b_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
										const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
										const bool a_conjugate, const bool b_conjugate, const size_t a_one, const size_t a_two,
										const size_t b_one, const size_t b_two, const size_t c_one, const size_t c_two,
										const tart::buffer_ptr& temp_buffer, const bool temp_buffer_provided,
										const Databases& db = Databases({"Gemm"}),
										tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);

	// Direct version of GEMM (no pre and post-processing kernels)
	void GemmDirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
									const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
									const size_t b_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
									const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
									const bool a_conjugate, const bool b_conjugate,
									const Databases& db = Databases({"Gemm"}),
									tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XGEMM_H_
#endif
