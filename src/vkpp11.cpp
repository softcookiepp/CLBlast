#include "vkpp11.hpp"

namespace clblast {
// =================================================================================================

static std::shared_ptr<tart::Instance> gInstance = nullptr;

tart::Instance& getInstance()
{
	if (!gInstance)
		gInstance = std::make_shared<tart::Instance>();
	return *gInstance;
}

// Represents a runtime error returned by an OpenCL API function
CLCudaAPIError::CLCudaAPIError(int32_t status, const std::string& where)
			: ErrorCode(status, where, "OpenCL error: " + where + ": " + std::to_string(static_cast<int>(status))) {}

void CLCudaAPIError::Check(const int32_t status, const std::string& where) {
		if (status != 0) {
			throw CLCudaAPIError(status, where);
		}
	}

void CLCudaAPIError::CheckDtor(const int32_t status, const std::string& where)
{
	if (status != 0) {
		fprintf(stderr, "CLBlast: %s (ignoring)\n", CLCudaAPIError(status, where).what());
	}
}

// Constructor based on the regular thingy
Device::Device(const tart::device_ptr device) : mDevice(device) {}

// Initialize the device. Note that this constructor can throw exceptions!
Device::Device(const size_t device_id) {
	// Use the global instance by default (this will mostly just be used for testing afaik)
	mDevice = getInstance().getDevice(device_id);
}

// Methods to retrieve device information
std::string Device::Version() const { return "Vulkan 1.2"; } // pretty sure this will work?
size_t Device::VersionNumber() const {
	return 120;
}
// TODO: implement some of this stuff in tart
std::string Device::Vendor() const
{
	switch(mDevice->getMetadata().physicalDeviceProperties.vendorID)
	{
	case tart::VendorID::eNVIDIA:
		return "NVIDIA";
	case tart::VendorID::eAMD:
		return "AMD";
	case tart::VendorID::eIntel:
		return "INTEL";
	default:
		return "Unknown vendor";
	}
}
std::string Device::Name() const { return "device name not implemented"; }
std::string Device::Type() const { return "GPU"; } // everything is a GPU when it comes to Vulkan! (for the most part)
size_t Device::MaxWorkGroupSize() const
{
	return mDevice->getMetadata().physicalDeviceProperties.limits.maxComputeWorkGroupInvocations;
}
std::vector<size_t> Device::MaxWorkItemSizes() const { return {1000000, 1000000, 1000000}; } // TODO: implement in Tart

// Not sure if Tart has a public method for querying extensions; might be a good idea to implement this.
std::string Device::Capabilities() const { return "not implemented"; }
bool Device::HasExtension(const std::string& extension) const
{
	// yeah, this doesn't work..
	return false;//return mDevice->supportsExtension(extension);
}

// Tart already has this
bool Device::SupportsFP64() const { return mDevice->getMetadata().double_; }
bool Device::SupportsFP16() const { return mDevice->getMetadata().half_; }
// Vulkan does not allow you to do this
size_t Device::CoreClock() const { return 0; }
// or this either.
size_t Device::ComputeUnits() const { return 0; }

// Vulkan has a way to do this, but I have been too lazy to implement it completely in Tart aside from error checking.
// Will have to do this eventually
unsigned long Device::MemorySize() const { return 0; }
// this can be retrieved from Tart, but may not be public
unsigned long Device::MaxAllocSize() const {
	return 0;
}

// Query for a specific type of device or brand
bool Device::IsCPU() const { return Type() == "CPU"; }
bool Device::IsGPU() const { return Type() == "GPU"; }

const RawDeviceID Device::operator()() const { return mDevice; }


// constructor for GLSL shaders
// requires multiple shader sources because each file can only have one entry point :c
Program::Program(const tart::device_ptr& device, std::map<std::string, std::string>& kernelSources)
{
	mProgramContainer = std::make_shared<tart::Program>(device, kernelSources);
}

// Accessor to the private data-member
std::shared_ptr<tart::Program> Program::operator()() const { return mProgramContainer; }

// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
Queue::Queue(const tart::device_ptr queue) { mDevice = queue; }

// Regular constructor with memory management
Queue::Queue(const Device& device)
{
	mDevice = device();
}

// Synchronizes the queue
void Queue::Finish() const { mDevice->sync(); }

// Retrieves the corresponding context or device
Device Queue::GetDevice() const {
	return Device(mDevice);
}

// Accessor to the private data-member
const RawCommandQueue& Queue::operator()() const { return mDevice; }







// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
template <typename T>
Buffer<T>::Buffer(const tart::buffer_ptr buffer) { buffer_ = buffer; }

// Regular constructor with memory management. If this class does not own the buffer object, then
// the memory will not be freed automatically afterwards. If the size is set to 0, this will
// become a stub containing a nullptr
template <typename T>
Buffer<T>::Buffer(const tart::device_ptr& device, const size_t size)
{
	if (size == 0)
		buffer_ = nullptr;
	else
		buffer_ = device->allocateBuffer(size*sizeof(T));
}

// Copies from device to host: reading the device buffer a-synchronously
// (this is currently impossible in tart, so it will just sync for now)
template <typename T>
void Buffer<T>::ReadAsync(const Queue& queue, const size_t size, T* host, const size_t offset) const
{
	if (offset > 0) throw LogicError("not implemented");
	buffer_->copyOut(host, size*sizeof(T), offset*sizeof(T));
}
template <typename T>
void Buffer<T>::ReadAsync(const Queue& queue, const size_t size, std::vector<T>& host, const size_t offset) const
{
	if (host.size() < size) {
		throw LogicError("Buffer: target host buffer is too small");
	}
	ReadAsync(queue, size, host.data(), offset);
}

// Copies from device to host: reading the device buffer
template <typename T>
void Buffer<T>::Read(const Queue& queue, const size_t size, T* host, const size_t offset) const {
	ReadAsync(queue, size, host, offset);
	queue.Finish();
}
template <typename T>
void Buffer<T>::Read(const Queue& queue, const size_t size, std::vector<T>& host, const size_t offset) const {
	Read(queue, size, host.data(), offset);
}

// Copies from host to device: writing the device buffer a-synchronously
template <typename T>
void Buffer<T>::WriteAsync(const Queue& queue, const size_t size, const T* host, const size_t offset) {
	if (GetSize() < (offset + size) * sizeof(T)) {
		throw LogicError("Buffer: target device buffer is too small");
	}
	if (offset > 0) throw LogicError("offsets greater than zero are not implemented :c");
	const void* hostbufVoid = host;
	void* hostptr = const_cast<void*>(hostbufVoid);
	buffer_->copyIn(hostptr, size*sizeof(T), offset*sizeof(T));
	//CheckError(clEnqueueWriteBuffer(queue(), *buffer_, CL_FALSE, offset * sizeof(T), size * sizeof(T), host, 0, nullptr,
	//																nullptr));
}
template <typename T>
void Buffer<T>::WriteAsync(const Queue& queue, const size_t size, const std::vector<T>& host, const size_t offset) {
	WriteAsync(queue, size, host.data(), offset);
}

// Copies from host to device: writing the device buffer
template <typename T>
void Buffer<T>::Write(const Queue& queue, const size_t size, const T* host, const size_t offset) {
	WriteAsync(queue, size, host, offset);
	queue.Finish();
}
template <typename T>
void Buffer<T>::Write(const Queue& queue, const size_t size, const std::vector<T>& host, const size_t offset) {
	Write(queue, size, host.data(), offset);
}

// Copies the contents of this buffer into another device buffer
template <typename T>
void Buffer<T>::CopyToAsync(const Queue& queue, const size_t size, const Buffer<T>& destination,
								 EventPointer event) const {
	if (event != nullptr) throw LogicError("copying with events is not implemented yet");
	buffer_->copyTo(destination(), 0, 0, size*sizeof(T));
	//CheckError(clEnqueueCopyBuffer(queue(), *buffer_, destination(), 0, 0, size * sizeof(T), 0, nullptr, event));
}
template <typename T>
void Buffer<T>::CopyTo(const Queue& queue, const size_t size, const Buffer<T>& destination) const {
	CopyToAsync(queue, size, destination);
	queue.Finish();
}

// Retrieves the actual allocated size in bytes
template <typename T>
size_t Buffer<T>::GetSize() const {
	return buffer_->getSize();
}

// Accessor to the private data-member
template <typename T>
tart::buffer_ptr Buffer<T>::operator()() const { return buffer_; }


template class Buffer<int8_t>;
template class Buffer<int16_t>;
template class Buffer<int32_t>;
template class Buffer<int64_t>;

template class Buffer<uint8_t>;
template class Buffer<uint16_t>;
template class Buffer<uint32_t>;
template class Buffer<uint64_t>;

template class Buffer<float>;
template class Buffer<double>;
template class Buffer<std::complex<float>>; // not using clblast::float2 because doing so requires circular includes that don't work
template class Buffer<std::complex<double>>; // same as above, but with clblast::double2







// difference between Vulkan and OpenCL as far as local sizes go will influence this outcome greatly...
Kernel::Kernel(const kernel_t kernel) { kernel_ = kernel; }
Kernel::Kernel(tart::program_ptr prg) { std::string ep = "none"; kernel_ = {ep, prg}; }

// Regular constructor with memory management
Kernel::Kernel(const std::shared_ptr<Program> program, const std::string& name)
{
	// this will be a bit different.
	// OpenCL allows kernels to be created that accept a variable local size.
	// Vulkan allows pipelines to be created where the entry point is specified,
	// but a fixed local size is used.
	// tart::CLProgram takes care of this, but it must be adapted to this library
	mEntryPoint = name;
	mProgramContainer = program->operator()();
	mDevice = mProgramContainer->getDevice();
	mKernel = mProgramContainer->getKernel(mEntryPoint);
}

// Retrieves the name of the kernel
std::string Kernel::GetFunctionName() const {
	return mEntryPoint;
}

// As above, but with an event waiting list
void Kernel::Launch(const Queue& queue, const std::vector<size_t>& global, const std::vector<size_t>& local)
{
	if (global.size() != local.size() ) throw LogicError("local and global size must be same length");
	std::vector<uint32_t> adjusted_global(global.size());
	for (size_t i = 0; i < global.size(); i += 1 )
	{
		if (global[i] % local[i] > 0) throw LogicError("global size must be divisible by local size");
		adjusted_global[i] = global[i] / local[i];
	}
	
	// convert local to uint32_t
	std::vector<uint32_t> local32(local.size());
	for (size_t i = 0; i < local.size(); i += 1)
	{
		local32[i] = local[i];
	}
	// ensure size is correct
	local32.resize(mKernel->getSpecConstantSize()/sizeof(uint32_t));
	
	mKernel->enqueue(adjusted_global, local32);
}

// Accessor to the private data-member
const kernel_t& Kernel::operator()() const { return kernel_; }


// =================================================================================================
}	// namespace clblast
