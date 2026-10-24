// no way...
#include "xgbmv.cpp"
#include "xgemv.cpp"
#include "xger.cpp"
#include "xgerc.cpp"
#include "xgeru.cpp"
#include "xhbmv.cpp"
#include "xhemv.cpp"
#include "xher.cpp"
#include "xher2.cpp"
#include "xhpmv.cpp"
#include "xhpr.cpp"
#include "xhpr2.cpp"
#include "xsbmv.cpp"
#include "xspmv.cpp"
#include "xspr.cpp"
#include "xspr2.cpp"
#include "xsymv.cpp"
#include "xsyr.cpp"
#include "xsyr2.cpp"
#include "xtbmv.cpp"
#include "xtpmv.cpp"
#include "xtrmv.cpp"
#include "xtrsv.cpp"

namespace clblast
{

void TestMatrixAP(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset)
{
	try
	{
		const auto required_size = (((n * (n + 1)) / 2) + offset) * dtype.size();
		if (buffer->getSize() < required_size)
		{
			throw BLASError(StatusCode::kInsufficientMemoryA);
		}
	}
	catch (const Error<std::runtime_error>& e)
	{
		throw BLASError(StatusCode::kInvalidMatrixA, e.what());
	}
}

} // namespace clblast
