// lets see if this reduces build time
#include "xamax.cpp"
#include "xasum.cpp"
#include "xaxpy.cpp"
#include "xcopy.cpp"
#include "xdot.cpp"
#include "xdotc.cpp"
#include "xdotu.cpp"
#include "xnrm2.cpp"
#include "xscal.cpp"
#include "xswap.cpp"

namespace clblast
{

// perfect place to put this too!
void TestVectorScalar(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset)
{
	try
	{
		const auto required_size = (n + offset) * dtype.size();
		if (buffer->getSize() < required_size) {
			throw BLASError(StatusCode::kInsufficientMemoryScalar);
		}
	}
	catch (const Error<std::runtime_error>& e)
	{
		throw BLASError(StatusCode::kInvalidVectorScalar, e.what());
	}
}


void TestVectorIndex(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset) {
	try {
		const auto required_size = (n + offset) * dtype.size();
		if (buffer->getSize() < required_size) {
			throw BLASError(StatusCode::kInsufficientMemoryScalar);
		}
	} catch (const Error<std::runtime_error>& e) {
		throw BLASError(StatusCode::kInvalidVectorScalar, e.what());
	}
}

// Tests vector 'X' for validity
void TestVectorX(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset, const size_t inc)
{
	if (inc == 0) {
		throw BLASError(StatusCode::kInvalidIncrementX);
	}
	try {
		const auto required_size = ((n - 1) * inc + 1 + offset) * dtype.size();
		if (buffer->getSize() < required_size) {
			throw BLASError(StatusCode::kInsufficientMemoryX);
		}
	} catch (const Error<std::runtime_error>& e) {
		throw BLASError(StatusCode::kInvalidVectorX, e.what());
	}
}

// Tests vector 'Y' for validity
void TestVectorY(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset, const size_t inc)
{
	if (inc == 0) {
		throw BLASError(StatusCode::kInvalidIncrementY);
	}
	try {
		const auto required_size = ((n - 1) * inc + 1 + offset) * dtype.size();
		if (buffer->getSize() < required_size) {
			throw BLASError(StatusCode::kInsufficientMemoryY);
		}
	} catch (const Error<std::runtime_error>& e) {
		throw BLASError(StatusCode::kInvalidVectorY, e.what());
	}
}

// Tests vector 'Z' for validity
void TestVectorZ(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset, const size_t inc)
{
	if (inc == 0) {
		throw BLASError(StatusCode::kInvalidIncrementZ);
	}
	try {
		const auto required_size = ((n - 1) * inc + 1 + offset) * dtype.size();
		if (buffer->getSize() < required_size) {
			throw BLASError(StatusCode::kInsufficientMemoryZ);
		}
	} catch (const Error<std::runtime_error>& e) {
		throw BLASError(StatusCode::kInvalidVectorZ, e.what());
	}
}

} // namespace clblast
