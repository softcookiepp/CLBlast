
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the common code to perform (partial) matrix inverting. This code is based
// on the TRSM implementation in the CUDA version of Magma version 2.2.0 and the poster "Triangular
// Linear System Solver for GPU with CUDA and OpenCL" by Peng Du, Stanimire Tomov, Piotr Luszczek,
// and Jack Dongarra.
//
// =================================================================================================

#include "routines/levelx/xinvert.hpp"

#include <assert.h>

#include <cstddef>
#include <string>
#include <vector>

#include "routine.hpp"
#include "routines/common.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
Xinvert::Xinvert(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name)
		: Routine(queue, event, name, {"Invert"}, dtype, {})
{
}

// =================================================================================================

// Inverts diagonal square blocks of a matrix
void Xinvert::InvertMatrixDiagonalBlocks(const Layout layout, const Triangle triangle, const Diagonal diag,
	const size_t n, const size_t block_size, const tart::buffer_ptr& src,
	const size_t offset, const size_t ld_src, tart::buffer_ptr& dest)
{
	// Makes sure all dimensions are larger than zero
	if ((block_size == 0) || (n == 0)) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Some parts of this kernel are not tunable and thus require some minimal OpenCL properties
	if (device_.MaxWorkGroupSize() < 16) {	// minimum of total local work size of 16
		throw RuntimeErrorCode(StatusCode::kNotImplemented);
	}

	// Helper variables
	const auto internal_block_size = static_cast<size_t>(db_["INTERNAL_BLOCK_SIZE"]);
	if (internal_block_size != 16) {
		throw RuntimeErrorCode(StatusCode::kNotImplemented);	// e.g. Apple CPU OpenCL with a WGS of 1
	}	// when barriers are present
	const auto num_blocks = CeilDiv(n, block_size);
	const auto num_internal_blocks = CeilDiv(n, internal_block_size);
	const auto unit_diagonal = (diag == Diagonal::kUnit) ? true : false;

	// This routine only supports block sizes which are a multiple of the internal block size and
	// block sizes up to and including 128
	if ((block_size % internal_block_size != 0) || (block_size > 128)) {
		throw BLASError(StatusCode::kUnknownError);
	}

	// Checks for validity of the source and destination matrices
	TestMatrixA(mDtype, n, n, src, offset, ld_src);
	TestMatrixB(mDtype, block_size, num_blocks * block_size, dest, 0, block_size);

	// Determines which kernels to run based on the layout (the kernels assume column-major as
	// default) and on whether we are dealing with an upper or lower triangle of the triangular matrix
	const bool is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
												 (triangle == Triangle::kLower && layout == Layout::kRowMajor));

	// Fills the output buffer with zeros
	FillMatrix(mDtype, program_, block_size,
						 num_blocks * block_size, block_size, 0, dest, tart::Scalar::zero(), 16);

	// Inverts the diagonal IB by IB inner blocks of the matrix: one block per work-group
	tart::kernel_ptr kernel = getInvertDiagonalBlock(this->mDevice, this->mDtype);
	kernel->setArg(0, static_cast<int>(n));
	kernel->setArg(1, src);
	kernel->setArg(2, static_cast<int>(offset));
	kernel->setArg(3, static_cast<int>(ld_src));
	kernel->setArg(4, dest);
	kernel->setArg(5, static_cast<int>(block_size));
	kernel->setArg(6, static_cast<int>(unit_diagonal));
	kernel->setArg(7, static_cast<int>(is_upper));
	const auto local_invert = std::vector<size_t>{internal_block_size};
	auto global_invert = std::vector<uint32_t>{num_internal_blocks * internal_block_size};
	global_invert[0] = global_invert[0] / local_invert[0];
	std::vector<uint32_t> spec = {
		db_["INTERNAL_BLOCK_SIZE"],
		db_["LOCALPAD"],
		db_["TMMWGSX"],
		db_["TMMWGSY"],
		0,
		static_cast<uint32_t>(is_upper)
	};
	kernel->enqueue(global_invert, spec);
	
	// Builds up block_size x block_size blocks. For example, internal_block_size=16:
	// use	 16 x 16	blocks to build	32 x 32	blocks,	1 x (1 x npages) grid,	4 x 4 threads;
	// then	32 x 32	blocks to build	64 x 64	blocks,	1 x (2 x npages) grid,	8 x 4 threads;
	// then	64 x 64	blocks to build 128 x 128 blocks,	1 x (4 x npages) grid, 16 x 4 threads;
	for (auto current_size = internal_block_size; current_size < block_size; current_size *= 2)
	{
		assert(current_size == 16 || current_size == 32 || current_size == 64);
		
		std::vector<uint32_t> spec2 = {
			db_["INTERNAL_BLOCK_SIZE"],
			db_["LOCALPAD"],
			db_["TMMWGSX"],
			db_["TMMWGSY"],
			current_size,
			static_cast<uint32_t>(is_upper)
		};

		// Emulates a 3D grid: NX * (NY * npages)
		const auto npages = CeilDiv(n, current_size * 2);
		const auto local0 = (current_size <= 32) ? current_size / 4 : 16;
		const auto local = std::vector<size_t>{local0, 4};
		auto global = std::vector<uint32_t>{Ceil(current_size / local[1], local[0]), Ceil(npages * (current_size / 16) * local[1], local[1])};
		global[0] = global[0] / local0;
		global[1] = global[1] / 4;

		// Part 1
		tart::kernel_ptr kernel1 = getInvertTripleMatmul(this->mDevice, this->mDtype);
		kernel1->setArg(0, static_cast<int>(n));
		kernel1->setArg(1, src);
		kernel1->setArg(2, static_cast<int>(offset));
		kernel1->setArg(3, static_cast<int>(ld_src));
		kernel1->setArg(4, dest);
		kernel1->setArg(5, static_cast<int>(current_size));
		kernel1->setArg(6, static_cast<int>(npages));
		kernel1->setArg(7, static_cast<int>(block_size));
		kernel1->enqueue(global, spec2);

		// Part 2
		tart::kernel_ptr kernel2 = getInvertTripleMatmulPart2(this->mDevice, this->mDtype);
		kernel2->setArg(0, static_cast<int>(n));
		kernel2->setArg(1, dest);
		kernel2->setArg(2, static_cast<int>(current_size));
		kernel2->setArg(3, static_cast<int>(npages));
		kernel2->setArg(4, static_cast<int>(block_size));
		kernel2->enqueue(global, spec2);
		
		// Exit in case we reach beyond the bounds of the input matrix
		if (current_size * 2 >= n) {
			break;
		}
	}
	
}

// =================================================================================================
}	// namespace clblast
