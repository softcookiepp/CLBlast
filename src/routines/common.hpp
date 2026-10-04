
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the interfaces to common kernels, such as copying, padding, and
// transposing a matrix. These functions are templated and thus header-only. This file also contains
// other common functions to routines, such as a function to launch a kernel.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_COMMON_H_
#define CLBLAST_ROUTINES_COMMON_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "database/database.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
void RunKernel(Kernel& kernel, Queue& queue, const Device& device, std::vector<size_t> global,
							 const std::vector<size_t>& local);

// =================================================================================================

// Sets all elements of a matrix to a constant value
template <typename T>
void FillMatrix(const std::shared_ptr<Program> program, const size_t m, const size_t n, const size_t ld,
								const size_t offset, const Buffer<T>& dest, const T constant_value, const size_t local_size);

// Sets all elements of a vector to a constant value
template <typename T>
void FillVector(const std::shared_ptr<Program> program, const size_t n, const size_t inc, const size_t offset,
								const Buffer<T>& dest, const T constant_value, const size_t local_size);

// =================================================================================================

// Copies or transposes a matrix and optionally pads/unpads it with zeros. This method is also able
// to write to symmetric and triangular matrices through optional arguments.
template <typename T>
void PadCopyTransposeMatrix(const Databases& db, 
														const size_t src_one, const size_t src_two,
														const size_t src_ld, const size_t src_offset, const Buffer<T>& src, const size_t dest_one,
														const size_t dest_two, const size_t dest_ld, const size_t dest_offset,
														const Buffer<T>& dest, const T alpha, const std::shared_ptr<Program>& program,
														const bool do_pad, const bool do_transpose, const bool do_conjugate,
														const bool upper = false, const bool lower = false, const bool diagonal_imag_zero = false)
{
	// Determines whether or not the fast-version could potentially be used
	auto use_fast_kernel = (src_offset == 0) && (dest_offset == 0) && (do_conjugate == false) && (src_one == dest_one) &&
												 (src_two == dest_two) && (src_ld == dest_ld) && (upper == false) && (lower == false) &&
												 (diagonal_imag_zero == false);

	// Determines the right kernel
	auto kernel_name = std::string{};
	auto pad_kernel = false;
	if (do_transpose)
	{
		if (use_fast_kernel && IsMultiple(src_ld, db["TRA_WPT"]) && IsMultiple(src_one, db["TRA_WPT"] * db["TRA_DIM"]) &&
				IsMultiple(src_two, db["TRA_WPT"] * db["TRA_DIM"]))
		{
			kernel_name = "TransposeMatrixFast";
		}
		else
		{
			use_fast_kernel = false;
			pad_kernel = (do_pad || do_conjugate);
			kernel_name = (pad_kernel) ? "TransposePadMatrix" : "TransposeMatrix";
		}
	}
	else
	{
		if (use_fast_kernel && IsMultiple(src_ld, db["COPY_VW"]) && IsMultiple(src_one, db["COPY_VW"] * db["COPY_DIMX"]) &&
				IsMultiple(src_two, db["COPY_WPT"] * db["COPY_DIMY"]))
		{
			kernel_name = "CopyMatrixFast";
		}
		else
		{
			use_fast_kernel = false;
			pad_kernel = do_pad;
			kernel_name = (pad_kernel) ? "CopyPadMatrix" : "CopyMatrix";
		}
	}

	// Retrieves the kernel from the compiled binary
	auto kernelOld = Kernel(program, kernel_name);
	tart::kernel_ptr kernel = kernelOld.get();
	
	// Sets the kernel arguments
	if (use_fast_kernel)
	{
		kernel->setArg(0, static_cast<int>(src_ld));
		kernel->setArg(1, src());
		kernel->setArg(2, dest());
		kernel->setArg(3, GetRealArg(alpha));
	} 
	else
	{
		kernel->setArg(0, static_cast<int>(src_one));
		kernel->setArg(1, static_cast<int>(src_two));
		kernel->setArg(2, static_cast<int>(src_ld));
		kernel->setArg(3, static_cast<int>(src_offset));
		kernel->setArg(4, src());
		kernel->setArg(5, static_cast<int>(dest_one));
		kernel->setArg(6, static_cast<int>(dest_two));
		kernel->setArg(7, static_cast<int>(dest_ld));
		kernel->setArg(8, static_cast<int>(dest_offset));
		kernel->setArg(9, dest());
		kernel->setArg(10, GetRealArg(alpha));
		if (pad_kernel)
		{
			kernel->setArg(11, static_cast<int>(do_conjugate));
		}
		else
		{
			kernel->setArg(11, static_cast<int>(upper));
			kernel->setArg(12, static_cast<int>(lower));
			kernel->setArg(13, static_cast<int>(diagonal_imag_zero));
		}
	}

	// Launches the kernel and returns the error code. Uses global and local thread sizes based on
	// parameters in the database.
	std::vector<uint32_t> global(3, 1);
	std::vector<uint32_t> local(3, 1);
	std::vector<uint32_t> spec = {
		db["COPY_DIMX"],
		db["COPY_DIMY"],
		db["COPY_WPT"],
		db["PAD_DIMX"],
		db["PAD_DIMY"],
		db["PAD_WPTX"],
		db["PAD_WPTY"],
		db["TRA_DIM"],
		db["TRA_PAD"],
		db["TRA_SHUFFLE"],
		db["PADTRA_TILE"],
		db["PADTRA_WPT"],
		db["PADTRA_PAD"]
	};
	
	if (do_transpose)
	{
		if (use_fast_kernel)
		{
			global = {dest_one / db["TRA_WPT"], dest_two / db["TRA_WPT"]};
			local = {db["TRA_DIM"], db["TRA_DIM"]};
		}
		else
		{
			global = {Ceil(CeilDiv(dest_one, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
																							Ceil(CeilDiv(dest_two, db["PADTRA_WPT"]), db["PADTRA_TILE"])};
			local = {db["PADTRA_TILE"], db["PADTRA_TILE"]};
		}
	}
	else
	{
		if (use_fast_kernel)
		{
			global = {dest_one / db["COPY_VW"], dest_two / db["COPY_WPT"]};
			local = {db["COPY_DIMX"], db["COPY_DIMY"]};
		} 
		else
		{
			global = {Ceil(CeilDiv(dest_one, db["PAD_WPTX"]), db["PAD_DIMX"]), Ceil(CeilDiv(dest_two, db["PAD_WPTY"]), db["PAD_DIMY"])};
			local = {db["PAD_DIMX"], db["PAD_DIMY"]};
		}
	}
	global.resize(3, 1);
	local.resize(3, 1);
	global[0] = global[0]/local[0];
	global[1] = global[1]/local[1];
	global[2] = global[2]/local[2];
	kernel->enqueue(global, spec);
}

// Batched version of the above
template <typename T>
void PadCopyTransposeMatrixBatched(const Databases& db, const size_t src_one, const size_t src_two,
																	 const size_t src_ld, const Buffer<int>& src_offsets, const Buffer<T>& src,
																	 const size_t dest_one, const size_t dest_two, const size_t dest_ld,
																	 const Buffer<int>& dest_offsets, const Buffer<T>& dest,
																	 const std::shared_ptr<Program>& program, const bool do_pad, const bool do_transpose,
																	 const bool do_conjugate, const size_t batch_count)
{
	// Determines the right kernel
	auto kernel_name = std::string{};
	if (do_transpose) {
		kernel_name = (do_pad) ? "TransposePadMatrixBatched" : "TransposeMatrixBatched";
	} else {
		kernel_name = (do_pad) ? "CopyPadMatrixBatched" : "CopyMatrixBatched";
	}

	// Retrieves the kernel from the compiled binary
	auto kernelOld = Kernel(program, kernel_name);
	tart::kernel_ptr kernel = kernelOld.get();

	// Sets the kernel arguments
	kernel->setArg(0, static_cast<int>(src_one));
	kernel->setArg(1, static_cast<int>(src_two));
	kernel->setArg(2, static_cast<int>(src_ld));
	kernel->setArg(3, src_offsets());
	kernel->setArg(4, src());
	kernel->setArg(5, static_cast<int>(dest_one));
	kernel->setArg(6, static_cast<int>(dest_two));
	kernel->setArg(7, static_cast<int>(dest_ld));
	kernel->setArg(8, dest_offsets());
	kernel->setArg(9, dest());
	if (do_pad) {
		kernel->setArg(10, static_cast<int>(do_conjugate));
	}

	// Launches the kernel and returns the error code. Uses global and local thread sizes based on
	// parameters in the database.
	std::vector<uint32_t> global(3, 1);
	std::vector<uint32_t> local(3, 1);
	std::vector<uint32_t> spec = {
		db["COPY_DIMX"],
		db["COPY_DIMY"],
		db["COPY_WPT"],
		db["PAD_DIMX"],
		db["PAD_DIMY"],
		db["PAD_WPTX"],
		db["PAD_WPTY"],
		db["TRA_DIM"],
		db["TRA_PAD"],
		db["TRA_SHUFFLE"],
		db["PADTRA_TILE"],
		db["PADTRA_WPT"],
		db["PADTRA_PAD"]
	};
	if (do_transpose) {
		global = {Ceil(CeilDiv(dest_one, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
																						Ceil(CeilDiv(dest_two, db["PADTRA_WPT"]), db["PADTRA_TILE"]), batch_count};
		local = {db["PADTRA_TILE"], db["PADTRA_TILE"], 1};
	} else {
		global = {Ceil(CeilDiv(dest_one, db["PAD_WPTX"]), db["PAD_DIMX"]),
																						Ceil(CeilDiv(dest_two, db["PAD_WPTY"]), db["PAD_DIMY"]), batch_count};
		local = {db["PAD_DIMX"], db["PAD_DIMY"], 1};
	}
	global.resize(3, 1);
	local.resize(3, 1);
	global[0] = global[0]/local[0];
	global[1] = global[1]/local[1];
	global[2] = global[2]/local[2];
	kernel->enqueue(global, spec);
}

// Batched version of the above
template <typename T>
void PadCopyTransposeMatrixStridedBatched(const Databases& db, const size_t src_one,
										const size_t src_two, const size_t src_ld, const size_t src_offset,
										const size_t src_stride, const Buffer<T>& src, const size_t dest_one,
										const size_t dest_two, const size_t dest_ld, const size_t dest_offset,
										const size_t dest_stride, const Buffer<T>& dest,
										const std::shared_ptr<Program>& program, const bool do_pad,
										const bool do_transpose, const bool do_conjugate, const size_t batch_count)
{
	// Determines the right kernel
	auto kernel_name = std::string{};
	if (do_transpose) {
		kernel_name = (do_pad) ? "TransposePadMatrixStridedBatched" : "TransposeMatrixStridedBatched";
	} else {
		kernel_name = (do_pad) ? "CopyPadMatrixStridedBatched" : "CopyMatrixStridedBatched";
	}

	// Retrieves the kernel from the compiled binary
	auto kernelOld = Kernel(program, kernel_name);
	tart::kernel_ptr kernel = kernelOld.get();
	
	// Sets the kernel arguments
	kernel->setArg(0, static_cast<int>(src_one));
	kernel->setArg(1, static_cast<int>(src_two));
	kernel->setArg(2, static_cast<int>(src_ld));
	kernel->setArg(3, static_cast<int>(src_offset));
	kernel->setArg(4, static_cast<int>(src_stride));
	kernel->setArg(5, src());
	kernel->setArg(6, static_cast<int>(dest_one));
	kernel->setArg(7, static_cast<int>(dest_two));
	kernel->setArg(8, static_cast<int>(dest_ld));
	kernel->setArg(9, static_cast<int>(dest_offset));
	kernel->setArg(10, static_cast<int>(dest_stride));
	kernel->setArg(11, dest());
	if (do_pad) {
		kernel->setArg(12, static_cast<int>(do_conjugate));
	}

	// Launches the kernel and returns the error code. Uses global and local thread sizes based on
	// parameters in the database.
	std::vector<uint32_t> global(3, 1);
	std::vector<uint32_t> local(3, 1);
	std::vector<uint32_t> spec = {
		db["COPY_DIMX"],
		db["COPY_DIMY"],
		db["COPY_WPT"],
		db["PAD_DIMX"],
		db["PAD_DIMY"],
		db["PAD_WPTX"],
		db["PAD_WPTY"],
		db["TRA_DIM"],
		db["TRA_PAD"],
		db["TRA_SHUFFLE"],
		db["PADTRA_TILE"],
		db["PADTRA_WPT"],
		db["PADTRA_PAD"]
	};
	if (do_transpose) {
		global = {Ceil(CeilDiv(dest_one, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
																						Ceil(CeilDiv(dest_two, db["PADTRA_WPT"]), db["PADTRA_TILE"]), batch_count};
		local = {db["PADTRA_TILE"], db["PADTRA_TILE"], 1};
	}
	else
	{
		global = {Ceil(CeilDiv(dest_one, db["PAD_WPTX"]), db["PAD_DIMX"]),
																						Ceil(CeilDiv(dest_two, db["PAD_WPTY"]), db["PAD_DIMY"]), batch_count};
		local = {db["PAD_DIMX"], db["PAD_DIMY"], 1};
	}
	global.resize(3, 1);
	local.resize(3, 1);
	global[0] = global[0]/local[0];
	global[1] = global[1]/local[1];
	global[2] = global[2]/local[2];
	kernel->enqueue(global, spec);
}

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_COMMON_H_
#endif
