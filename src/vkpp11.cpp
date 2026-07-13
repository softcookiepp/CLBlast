#include "vkpp11.hpp"

namespace clblast {
// =================================================================================================

// Represents a runtime error returned by an OpenCL API function
CLCudaAPIError::CLCudaAPIError(int32_t status, const std::string& where)
			: ErrorCode(status, where, "OpenCL error: " + where + ": " + std::to_string(static_cast<int>(status))) {}

void CLCudaAPIError::Check(const int32_t status, const std::string& where) {
		if (status != CL_SUCCESS) {
			throw CLCudaAPIError(status, where);
		}
	}

void CLCudaAPIError::CheckDtor(const int32_t status, const std::string& where)
{
	if (status != CL_SUCCESS) {
		fprintf(stderr, "CLBlast: %s (ignoring)\n", CLCudaAPIError(status, where).what());
	}
}

	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
Event::Event(const tart::event_ptr event) { mEvent = event; }

	// Regular constructor with memory management
Event::Event()
{
	mEvent = std::make_shared<tart::Event>();
}

// Waits for completion of this event
void Event::WaitForCompletion() const
{
	if (mEvent && mEvent->isActive()) mEvent->sync();
}

// Retrieves the elapsed time of the last recorded event.
// (Note that there is a bug in Apple's OpenCL implementation of the 'clGetEventProfilingInfo' function:
//	http://stackoverflow.com/questions/26145603/clgeteventprofilinginfo-bug-in-macosx)
// However, in our case the reply size is fixed to be uint64_t, so we are not affected.
float Event::GetElapsedTime() const
{
	// not implemented yet :c
	WaitForCompletion();
	return 1.0;
}

// Accessor to the private data-member
tart::event_ptr Event::operator()() { return mEvent; }
const tart::event_ptr Event::operator()() const { return mEvent; }
tart::event_ptr  Event::pointer() { return mEvent; }
const tart::event_ptr Event::pointer() const { return mEvent; }

// Initializes the platform
Platform::Platform(const size_t platform_id)
{
	// there can only be one, this is Vulkan c:
	if (platform_id != 0) throw LogicError("Vulkan back-end requires a platform ID of 0");
	mInstance = std::make_shared<tart::Instance>();
}

// Methods to retrieve platform information
std::string Platform::Name() const {
	return "not implemented";
}
std::string Platform::Vendor() const { return "not implemented"; }
std::string Platform::Version() const { return "not implemented"; }
	
// returns the tart::Instance
tart::Instance& Platform::getInstance() const
{
	return *mInstance;
}

// Returns the number of devices on this platform
size_t Platform::NumDevices()
{
	return static_cast<size_t>(getInstance().getNumDevices());
}

// Accessor to the private data-member
const RawPlatformID& Platform::operator()() const { return platform_; }



// Constructor based on the regular thingy
Device::Device(const tart::device_ptr device) : mDevice(device) {}

// Initialize the device. Note that this constructor can throw exceptions!
Device::Device(const Platform& platform, const size_t device_id) {
	// Use the global instance by default (this will mostly just be used for testing afaik)
	mDevice = platform.getInstance().getDevice(device_id);
}

// Methods to retrieve device information
// (platform id is always 0)
RawPlatformID Device::PlatformID() const { return 0; }
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
size_t Device::MaxWorkGroupSize() const { return 1000000; } // straight-up no idea how to even go about doing this. in Vulkan, each dimension can be different.
size_t Device::MaxWorkItemDimensions() const { return 3; } // it is always 3 in vulkan
std::vector<size_t> Device::MaxWorkItemSizes() const { return {1000000, 1000000, 1000000}; } // TODO: implement in Tart
unsigned long Device::LocalMemSize() const
{
#if 1
	// TODO: actually implement in Tart
	return 0;
#else
	return static_cast<unsigned long>(GetInfo<uint64_t>(CL_DEVICE_LOCAL_MEM_SIZE));
#endif
}

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

// neither of these can be queried in Vulkan either
size_t Device::MemoryClock() const { return 0; }		 // Not exposed in OpenCL
size_t Device::MemoryBusWidth() const { return 0; }	// Not exposed in OpenCL

// Configuration-validity checks
bool Device::IsLocalMemoryValid(const uint64_t local_mem_usage) const
{
#if 1
	// not yet implemented
	return true;
#else
	return (local_mem_usage <= LocalMemSize());
#endif
}
bool Device::IsThreadConfigValid(const std::vector<size_t>& local) const { return true; }

// Query for a specific type of device or brand
bool Device::IsCPU() const { return Type() == "CPU"; }
bool Device::IsGPU() const { return Type() == "GPU"; }
bool Device::IsAMD() const
{
	return Device::Vendor() == "AMD";
}
bool Device::IsNVIDIA() const
{
	return false;
}
bool Device::IsIntel() const
{
	return Vendor() == "INTEL";
}
bool Device::IsARM() const
{
	return Device::Vendor() == "ARM";
}
bool Device::IsQualcomm() const
{
	return Device::Vendor() == "Qualcomm";
}

std::string Device::AMDBoardName() const
{
	return "not implemented";
}
std::string Device::NVIDIAComputeCapability() const {	// check for 'cl_nv_device_attribute_query' first
	// dummy
	return "SM3.7";
}

bool Device::IsPostNVIDIAVolta() const
{
	return false;
}

std::string Device::AdrenoVersion() const
{
	return "not implemented";
}

std::string Device::GetExtraInfo() const
{
	return "not implemented";
}

const RawDeviceID Device::operator()() const { return mDevice; }

using RawContext = tart::device_ptr;

// Constructor based on tart::device_ptr
Context::Context(tart::device_ptr context) { mDevice = context; }

// Regular constructor with memory management
Context::Context(const Device& device)
{
	mDevice = device();
}

// Accessor to the private data-member
const RawContext Context::operator()() const { return mDevice; }
RawContext Context::pointer() const { return mDevice; }


// Source-based constructor with memory management
Program::Program(const clblast::Context& context, const std::string& source)
{
	mSource = source;
	mDevice = context.pointer();
}

// constructor for GLSL shaders
// requires multiple shader sources because each file can only have one entry point :c
Program::Program(const clblast::Context& context, std::map<std::string, std::string>& kernelSources):
	mDevice(context.pointer())
{
	mProgramContainer = std::make_shared<tart::Program>(context.pointer(), kernelSources);
}

// Binary-based constructor with memory management
Program::Program(const clblast::Device& device, const clblast::Context& context, std::string& binary) {
	mDevice = context.pointer();
	
	std::cout << "BINARY STRING:\n\n" << binary << "\n\n\n";
	throw LogicError("Construction from string not implement yet; may need to reinterpret as spv or something");
	
	//mShaderModule = mDevice->loadShader(binary);
	//mCLProgram = mDevice->createCLProgram(mShaderModule);
}

// Compiles the device program and checks whether or not there are any warnings/errors
void Program::Build(const clblast::Device& device, const clblast::Context& context, std::vector<std::string>& options) {
	Build(device, options);
}

// Compiles the device program and checks whether or not there are any warnings/errors
void Program::Build(const clblast::Device& device, std::vector<std::string>& options) {
	// if program container already here, don't bother
	if (mProgramContainer) return;
	throw std::runtime_error("who knows lool");
	// TODO: parse options (look for dflags, etc.)
	// compile the shader module
	// mShaderModule = mDevice->compileCL(mSource);
	// load it into the actual CL program handler thingy
	// mProgramContainer = std::make_shared<tart::Program>(mDevice, mDevice->createCLProgram(mShaderModule));
	//mCLProgram = mDevice->createCLProgram(mShaderModule);
}

// Confirms whether a certain status code is an actual compilation error or warning
bool Program::StatusIsCompilationWarningOrError(const int32_t status) const { return (status == -11); }

// Retrieves the warning/error message from the compiler (if any)
std::string Program::GetBuildInfo(const clblast::Device& device) const {
	return "not implemented yet :c";
}

// Retrieves a binary or an intermediate representation of the compiled program
std::string Program::GetIR() const {
	// TODO: somehow convert the SPIR-V into a string?
	return "not implemented yet :c";
}

// Accessor to the private data-member
std::shared_ptr<tart::Program> Program::operator()() const { return mProgramContainer; }

// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
Queue::Queue(const tart::device_ptr queue) { mDevice = queue; }

// Regular constructor with memory management
Queue::Queue(const Context& context, const Device& device)
{
	mDevice = context.pointer();
}

// Synchronizes the queue
void Queue::Finish(Event& event) const { mDevice->sync({event.pointer()}); }
void Queue::Finish() const { mDevice->sync(); }

// Retrieves the corresponding context or device
Context Queue::GetContext() const {
	return Context(mDevice);
}
Device Queue::GetDevice() const {
	return Device(mDevice);
}

// Accessor to the private data-member
const RawCommandQueue& Queue::operator()() const { return mDevice; }







// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
template <typename T>
Buffer<T>::Buffer(const tart::buffer_ptr buffer) : access_(BufferAccess::kNotOwned) { buffer_ = buffer; }

// Regular constructor with memory management. If this class does not own the buffer object, then
// the memory will not be freed automatically afterwards. If the size is set to 0, this will
// become a stub containing a nullptr
template <typename T>
Buffer<T>::Buffer(const Context& context, const BufferAccess access, const size_t size) :
			access_(access)
{
	if (size == 0)
		buffer_ = nullptr;
	else
		buffer_ = context.pointer()->allocateBuffer(size*sizeof(T));
}

// As above, but now with read/write access as a default
template <typename T>
Buffer<T>::Buffer(const Context& context, const size_t size) : Buffer<T>(context, BufferAccess::kReadWrite, size) {}

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
template <typename T>
void Buffer<T>::ReadAsync(const Queue& queue, const size_t size, BufferHost<T>& host, const size_t offset) const {
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
template <typename T>
void Buffer<T>::Read(const Queue& queue, const size_t size, BufferHost<T>& host, const size_t offset) const {
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
template <typename T>
void Buffer<T>::WriteAsync(const Queue& queue, const size_t size, const BufferHost<T>& host, const size_t offset) {
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
template <typename T>
void Buffer<T>::Write(const Queue& queue, const size_t size, const BufferHost<T>& host, const size_t offset) {
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

// Retrieves the amount of local memory used per work-group for this kernel
unsigned long Kernel::LocalMemUsage(const Device& device) const {
	// It doesn't seem that Vulkan has a direct equivalent to this.
	// More investigation will have to be done.
	// It will likely have something to with SPIR-V reflection...
	return 0;
}

// Retrieves the name of the kernel
std::string Kernel::GetFunctionName() const {
	return mEntryPoint;
}

// As above, but with an event waiting list
void Kernel::Launch(const Queue& queue, const std::vector<size_t>& global, const std::vector<size_t>& local,
						EventPointer event, const std::vector<Event>& waitForEvents, const tart::command_sequence_ptr& sequence)
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
	
	if (sequence)
	{
		// record to sequence and submit later
		mKernel->record(sequence, adjusted_global, local32);
	}
	else
	{
		std::vector<tart::event_ptr> wait(waitForEvents.size(), nullptr);
		for (size_t i = 0; i < waitForEvents.size(); i += 1)
		{
			wait[i] = waitForEvents[i].pointer();
		}
		
		mKernel->enqueue(adjusted_global, local32, wait);
	}
}

// Accessor to the private data-member
const kernel_t& Kernel::operator()() const { return kernel_; }


// =================================================================================================
}	// namespace clblast
