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

} // namespace clblast
