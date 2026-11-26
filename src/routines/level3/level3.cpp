// it really works!
#include "routine_aot_cache.hpp"
#include "xgemm.cpp"
#include "xhemm.cpp"
#include "xher2k.cpp"
#include "xherk.cpp"
#include "xsymm.cpp"
#include "xsyr2k.cpp"
#include "xsyrk.cpp"
#include "xtrmm.cpp"
#include "xtrsm.cpp"

namespace clblast
{

void TestMatrixA(const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const uint32_t offset, const uint32_t ld, const bool test_lead_dim)
{
	if (test_lead_dim && ld < one)
	{
		throw BLASError(StatusCode::kInvalidLeadDimA);
	}
	try {
		const auto required_size = (ld * (two - 1) + one + offset) * dtype.size();
		if (buffer->getSize() < required_size) {
			throw BLASError(StatusCode::kInsufficientMemoryA);
		}
	} catch (const Error<std::runtime_error>& e) {
		throw BLASError(StatusCode::kInvalidMatrixA, e.what());
	}
}

void TestMatrixB(const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const uint32_t offset, const uint32_t ld, const bool test_lead_dim)
{
	if (test_lead_dim && ld < one)
	{
		throw BLASError(StatusCode::kInvalidLeadDimB);
	}
	try {
		const auto required_size = (ld * (two - 1) + one + offset) * dtype.size();
		if (buffer->getSize() < required_size) {
			throw BLASError(StatusCode::kInsufficientMemoryB);
		}
	} catch (const Error<std::runtime_error>& e) {
		throw BLASError(StatusCode::kInvalidMatrixB, e.what());
	}
}

void TestMatrixC(const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const uint32_t offset, const uint32_t ld, const bool test_lead_dim)
{
	if (test_lead_dim && ld < one)
	{
		throw BLASError(StatusCode::kInvalidLeadDimC);
	}
	try {
		const auto required_size = (ld * (two - 1) + one + offset) * dtype.size();
		if (buffer->getSize() < required_size) {
			throw BLASError(StatusCode::kInsufficientMemoryC);
		}
	} catch (const Error<std::runtime_error>& e) {
		throw BLASError(StatusCode::kInvalidMatrixC, e.what());
	}
}

} // namespace clblast
