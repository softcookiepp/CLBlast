
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a bunch of C++11 classes that act as wrappers around OpenCL objects and API
// calls. The main benefits are increased abstraction, automatic memory management, and portability.
// Portability here means that a similar header exists for CUDA with the same classes and
// interfaces. In other words, moving from the OpenCL API to the CUDA API becomes a one-line change.
//
// This file is taken from the CLCudaAPI project <https://github.com/CNugteren/CLCudaAPI> and
// therefore contains the following header copyright notice:
//
// =================================================================================================
//
// Copyright 2015 SURFsara
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =================================================================================================

// IWYU pragma: private, include "utilities/backend.hpp"

#ifndef CLBLAST_VKPP11_H_
#define CLBLAST_VKPP11_H_

// C++
#include <assert.h>

#include <algorithm>	// std::copy
#include <cstdio>		 // fprintf, stderr
#include <cstring>		// std::strlen
#include <memory>		 // std::shared_ptr
#include <numeric>		// std::accumulate
#include <string>		 // std::string
#include <vector>		 // std::vector
#include <map>


// Android support (missing C++11 functions to_string, stod, and stoi)
#ifdef __ANDROID__
#include "utilities/android.hpp"	// IWYU pragma: export
#endif

// Exception classes
#include "cxpp11_common.hpp"

#ifndef CL_TEMP_DEFS
#define CL_DEMP_DEFS
#define CL_SUCCESS 0
#endif
#include "tart.hpp"

namespace clblast {
// =================================================================================================

// Represents a runtime error returned by an OpenCL API function
class CLCudaAPIError : public ErrorCode<DeviceError, int32_t> {
public:
	explicit CLCudaAPIError(int32_t status, const std::string& where)
			: ErrorCode(status, where, "OpenCL error: " + where + ": " + std::to_string(static_cast<int>(status))) {}

	static void Check(const int32_t status, const std::string& where) {
		if (status != CL_SUCCESS) {
			throw CLCudaAPIError(status, where);
		}
	}

	static void CheckDtor(const int32_t status, const std::string& where) {
		if (status != CL_SUCCESS) {
			fprintf(stderr, "CLBlast: %s (ignoring)\n", CLCudaAPIError(status, where).what());
		}
	}
};

// Exception returned when building a program
using CLCudaAPIBuildError = CLCudaAPIError;

// =================================================================================================

// Error occurred in OpenCL
#define CheckError(call) CLCudaAPIError::Check(call, CLCudaAPIError::TrimCallString(#call))

// Error occurred in OpenCL (no-exception version for destructors)
#define CheckErrorDtor(call) CLCudaAPIError::CheckDtor(call, CLCudaAPIError::TrimCallString(#call))

// =================================================================================================

// pointer to event
typedef tart::event_ptr EventPointer;

// C++11 version of 'cl_event'
class Event {
	// An event, in regards to Tart compatibility (for now)
	// will just be a list of sequences that the device will away the completion of.
	// which means Tart must be modified to be able to get the parent device from a sequence?
	// ugh
	tart::event_ptr mEvent = nullptr;
public:

	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
	explicit Event(const tart::event_ptr event) { mEvent = event; }

	// Regular constructor with memory management
	explicit Event()
	{
		// ok, we are actually going to want to change this
		// the problem is that 
		mEvent = std::make_shared<tart::Event>(nullptr);
	}

	// Waits for completion of this event
	void WaitForCompletion() const
	{
#if 1
		// not going to bother with this now
		//sequence->getDevice()->sync();
#else
		if (mEvent == nullptr)
			throw LogicError("Sequence cannot be null!");
		auto sequence = mEvent->getSequence();
		sequence->getDevice()->sync({sequence});
#endif
	}

	// Retrieves the elapsed time of the last recorded event.
	// (Note that there is a bug in Apple's OpenCL implementation of the 'clGetEventProfilingInfo' function:
	//	http://stackoverflow.com/questions/26145603/clgeteventprofilinginfo-bug-in-macosx)
	// However, in our case the reply size is fixed to be uint64_t, so we are not affected.
	float GetElapsedTime() const {
#if 1
		// not implemented yet :c
		return 1.0;
#else
		WaitForCompletion();
		const auto bytes = sizeof(uint64_t);
		auto time_start = uint64_t{0};
		CheckError(clGetEventProfilingInfo(*event_, CL_PROFILING_COMMAND_START, bytes, &time_start, nullptr));
		auto time_end = uint64_t{0};
		CheckError(clGetEventProfilingInfo(*event_, CL_PROFILING_COMMAND_END, bytes, &time_end, nullptr));
		return static_cast<float>(time_end - time_start) * 1.0e-6f;
#endif
	}

	// Accessor to the private data-member
	tart::event_ptr operator()() { return mEvent; }
	const tart::event_ptr operator()() const { return mEvent; }
	tart::event_ptr  pointer() { return mEvent; }
	const tart::event_ptr pointer() const { return mEvent; }
};

#if 1
// =================================================================================================

// Raw platform ID type
// since Vulkan doesn't have platforms like OpenCL does, there will only be a single platform that has all the devices.
using RawPlatformID = size_t;

// Vulkan doesn't have any direct equivalent to this, just use it to encapsulate device..or maybe instance?
class Platform {
public:
	// Initializes the platform
	explicit Platform(const size_t platform_id)
	{
		// there can only be one, this is Vulkan c:
		if (platform_id != 0) throw LogicError("Vulkan back-end requires a platform ID of 0");
	}

	// Methods to retrieve platform information
	std::string Name() const {
#if 1
		return "not implemented";
#else
		return GetInfoString(CL_PLATFORM_NAME);
#endif
	}
	std::string Vendor() const { return "not implemented"; }
	std::string Version() const { return "not implemented"; }

	// Returns the number of devices on this platform
	size_t NumDevices()
	{
		return static_cast<size_t>(tart::init().getNumDevices());
	}

	// Accessor to the private data-member
	const RawPlatformID& operator()() const { return platform_; }
	tart::Instance& getInstance() { return tart::init(); }

private:
	size_t platform_ = 0;

	// Private helper functions
	std::string GetInfoString(const size_t info) const {
		// no idea what this is supposed to do; we find out L A T E R
		return "not implemented";
	}
};
#endif

#if 1
// not applicable for vulkan
// Retrieves a vector with all platforms
inline std::vector<Platform> GetAllPlatforms() {
#if 1
	// TODO: populate this based on single vulkan devices.
	std::vector<Platform> platforms({Platform(0)});
	return platforms;
#else
	auto num_platforms = cl_uint{0};
	CheckError(clGetPlatformIDs(0, nullptr, &num_platforms));
	auto all_platforms = std::vector<Platform>();
	for (size_t platform_id = 0; platform_id < static_cast<size_t>(num_platforms); ++platform_id) {
		all_platforms.push_back(Platform(platform_id));
	}
	return all_platforms;
#endif
}
#endif

#if 1
// =================================================================================================
// Raw device ID type
using RawDeviceID = tart::device_ptr;

// Tart already has a base device class that already covers a lot of this.
// Lets seee..............
// C++11 version of 'cl_device_id'
class Device {
	tart::device_ptr mDevice;
public:
	// Constructor based on the regular thingy
	explicit Device(const tart::device_ptr device) : mDevice(device) {}

	// Initialize the device. Note that this constructor can throw exceptions!
	explicit Device(const Platform& platform, const size_t device_id) {
		// Use the global instance by default (this will mostly just be used for testing afaik)
		mDevice = tart::init().createDevice(device_id);
	}

	// Methods to retrieve device information
	// (platform id is always 0)
	RawPlatformID PlatformID() const { return 0; }
	std::string Version() const { return "Vulkan 1.2"; } // pretty sure this will work?
	size_t VersionNumber() const {
		return 120;
	}
	// TODO: implement some of this stuff in tart
	std::string Vendor() const
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
	std::string Name() const { return "device name not implemented"; }
	std::string Type() const { return "GPU"; } // everything is a GPU when it comes to Vulkan! (for the most part)
	size_t MaxWorkGroupSize() const { return 1000000; } // straight-up no idea how to even go about doing this. in Vulkan, each dimension can be different.
	size_t MaxWorkItemDimensions() const { return 3; } // it is always 3 in vulkan
	std::vector<size_t> MaxWorkItemSizes() const { return {1000000, 1000000, 1000000}; } // TODO: implement in Tart
	unsigned long LocalMemSize() const
	{
#if 1
		// TODO: actually implement in Tart
		return 0;
#else
		return static_cast<unsigned long>(GetInfo<uint64_t>(CL_DEVICE_LOCAL_MEM_SIZE));
#endif
	}

	// Not sure if Tart has a public method for querying extensions; might be a good idea to implement this.
	std::string Capabilities() const { return "not implemented"; }
	bool HasExtension(const std::string& extension) const
	{
		// yeah, this doesn't work..
		return false;//return mDevice->supportsExtension(extension);
	}
	
	// Tart already has this
	bool SupportsFP64() const { return mDevice->getMetadata().double_; }
	bool SupportsFP16() const { return mDevice->getMetadata().half_; }
	// Vulkan does not allow you to do this
	size_t CoreClock() const { return 0; }
	// or this either.
	size_t ComputeUnits() const { return 0; }
	
	// Vulkan has a way to do this, but I have been too lazy to implement it completely in Tart aside from error checking.
	// Will have to do this eventually
	unsigned long MemorySize() const { return 0; }
	// this can be retrieved from Tart, but may not be public
	unsigned long MaxAllocSize() const {
		return 0;
	}
	
	// neither of these can be queried in Vulkan either
	size_t MemoryClock() const { return 0; }		 // Not exposed in OpenCL
	size_t MemoryBusWidth() const { return 0; }	// Not exposed in OpenCL

	// Configuration-validity checks
	bool IsLocalMemoryValid(const uint64_t local_mem_usage) const
	{
#if 1
		// not yet implemented
		return true;
#else
		return (local_mem_usage <= LocalMemSize());
#endif
	}
	bool IsThreadConfigValid(const std::vector<size_t>& local) const {
#if 1
#else
		auto local_size = size_t{1};
		for (const auto& item : local) {
			local_size *= item;
		}
		for (auto i = size_t{0}; i < local.size(); ++i) {
			if (local[i] > MaxWorkItemSizes()[i]) {
				return false;
			}
		}
		if (local_size > MaxWorkGroupSize()) {
			return false;
		}
		if (local.size() > MaxWorkItemDimensions()) {
			return false;
		}
		return true;
#endif
	}

	// Query for a specific type of device or brand
	bool IsCPU() const { return Type() == "CPU"; }
	bool IsGPU() const { return Type() == "GPU"; }
	bool IsAMD() const
	{
		return Vendor() == "AMD";
	}
	bool IsNVIDIA() const
	{
		// ugh, stupid assumptions about subgroup this and that...
		return false;
		//return Vendor() == "NVIDIA";
	}
	bool IsIntel() const
	{
		return Vendor() == "INTEL";
	}
	bool IsARM() const
	{
		return Vendor() == "ARM";
	}
	bool IsQualcomm() const
	{
		return Vendor() == "Qualcomm";
	}

	// Platform specific extensions
	std::string AMDBoardName() const
	{
		return "not implemented";
	}
	std::string NVIDIAComputeCapability() const {	// check for 'cl_nv_device_attribute_query' first
		// dummy
		return "SM3.7";
	}

	// Returns if the Nvidia chip is a Volta or later archicture (sm_70 or higher)
	bool IsPostNVIDIAVolta() const
	{
		return false;
	}

	// Returns the Qualcomm Adreno GPU version (i.e. a650, a730, a740, etc.)
	std::string AdrenoVersion() const
	{
		return "not implemented";
	}

	// Retrieves the above extra information (if present)
	std::string GetExtraInfo() const
	{
		return "not implemented";
	}

	// Accessor to the private data-member
	const RawDeviceID operator()() const { return mDevice; }
	
private:
#if 0
	// Private helper functions
	template <typename T>
	T GetInfo(const size_t info) const {
		auto bytes = size_t{0};
		CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
		auto result = T(0);
		CheckError(clGetDeviceInfo(device_, info, bytes, &result, nullptr));
		return result;
	}
	template <typename T>
	std::vector<T> GetInfoVector(const size_t info) const {
		auto bytes = size_t{0};
		CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
		auto result = std::vector<T>(bytes / sizeof(T));
		CheckError(clGetDeviceInfo(device_, info, bytes, result.data(), nullptr));
		return result;
	}
	std::string GetInfoString(const size_t info) const {
		auto bytes = size_t{0};
		CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
		auto result = std::string{};
		result.resize(bytes);
		CheckError(clGetDeviceInfo(device_, info, bytes, &result[0], nullptr));
		result.resize(strlen(result.c_str()));	// Removes any trailing '\0'-characters
		return result;
	}
#endif
};
#endif
// =================================================================================================

// ok, so the `Device` is more like `vk::PhysicalDevice` and the `Context` is more akin to `vk::Device`
// that makes sense.
#if 1
// Raw context type
using RawContext = tart::device_ptr;

// C++11 version of 'cl_context'
class Context {
	tart::device_ptr mDevice = nullptr;

public:
	// Constructor based on tart::device_ptr
	explicit Context(tart::device_ptr context) { mDevice = context; }

	// Regular constructor with memory management
	explicit Context(const Device& device)
	{
		mDevice = device();
	}

	// Accessor to the private data-member
	const RawContext operator()() const { return mDevice; }
	RawContext pointer() const { return mDevice; }

//private:
	//std::shared_ptr<cl_context> context_;
};
#endif
#if 1
// Pointer to an OpenCL context
using ContextPointer = tart::device_ptr;
// =================================================================================================

// C++11 version of 'cl_program'.
class Program {
	std::string mSource;
	tart::device_ptr mDevice = nullptr;
	std::shared_ptr<tart::Program> mProgramContainer = nullptr;
	tart::shader_module_ptr mShaderModule = nullptr;
	//tart::cl_program_ptr mCLProgram = nullptr;
public:
	// Source-based constructor with memory management
	explicit Program(const clblast::Context& context, const std::string& source)
	{
		mSource = source;
		mDevice = context.pointer();
	}
	
	// constructor for GLSL shaders
	// requires multiple shader sources because each file can only have one entry point :c
	explicit Program(const clblast::Context& context, std::map<std::string, std::string>& kernelSources):
		mDevice(context.pointer())
	{
		mProgramContainer = std::make_shared<tart::Program>(context.pointer(), kernelSources);
	}

	// Binary-based constructor with memory management
	explicit Program(const clblast::Device& device, const clblast::Context& context, std::string& binary) {
		mDevice = context.pointer();
		
		std::cout << "BINARY STRING:\n\n" << binary << "\n\n\n";
		throw LogicError("Construction from string not implement yet; may need to reinterpret as spv or something");
		
		//mShaderModule = mDevice->loadShader(binary);
		//mCLProgram = mDevice->createCLProgram(mShaderModule);
	}

	// Compiles the device program and checks whether or not there are any warnings/errors
	void Build(const clblast::Device& device, const clblast::Context& context, std::vector<std::string>& options) {
		Build(device, options);
	}
	
	// Compiles the device program and checks whether or not there are any warnings/errors
	void Build(const clblast::Device& device, std::vector<std::string>& options) {
		// if program container already here, don't bother
		if (mProgramContainer) return;
		
		// TODO: parse options (look for dflags, etc.)
		// compile the shader module
		mShaderModule = mDevice->compileCL(mSource);
		// load it into the actual CL program handler thingy
		mProgramContainer = std::make_shared<tart::Program>(mDevice, mDevice->createCLProgram(mShaderModule));
		//mCLProgram = mDevice->createCLProgram(mShaderModule);
	}

	// Confirms whether a certain status code is an actual compilation error or warning
	bool StatusIsCompilationWarningOrError(const int32_t status) const { return (status == -11); }

	// Retrieves the warning/error message from the compiler (if any)
	std::string GetBuildInfo(const clblast::Device& device) const {
		return "not implemented yet :c";
	}

	// Retrieves a binary or an intermediate representation of the compiled program
	std::string GetIR() const {
		// TODO: somehow convert the SPIR-V into a string?
		return "not implemented yet :c";
	}

	// Accessor to the private data-member
	std::shared_ptr<tart::Program> operator()() const { return mProgramContainer; }
};
#endif

// =================================================================================================
// Tart does not expose queues (yet?)
// since each device only has a single compute queue
// Raw command-queue type
using RawCommandQueue = tart::device_ptr;
#if 1
// no idea how to handle this, since Tart uses a single queue
// pretty sure I will just end up scrapping it, since tart::Device handles all this already (actually I can't)
// C++11 version of 'cl_command_queue'
class Queue {
	tart::device_ptr mDevice = nullptr;
public:
	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
	explicit Queue(const tart::device_ptr queue) { mDevice = queue; }

	// Regular constructor with memory management
	explicit Queue(const Context& context, const Device& device)
	{
		mDevice = context.pointer();
	}

	// Synchronizes the queue
	void Finish(Event&) const { Finish(); }
	void Finish() const { mDevice->sync(); }

	// Retrieves the corresponding context or device
	Context GetContext() const {
		return Context(mDevice);
	}
	Device GetDevice() const {
		return Device(mDevice);
	}

	// Accessor to the private data-member
	const RawCommandQueue& operator()() const { return mDevice; }
};
#endif
// =================================================================================================
#if 1
// C++11 version of host memory
template <typename T>
class BufferHost {

public:
	// Regular constructor with memory management
	explicit BufferHost(const Context&, const size_t size) : buffer_(new std::vector<T>(size)) {}

	// Retrieves the actual allocated size in bytes
	size_t GetSize() const { return buffer_->size() * sizeof(T); }

	// Compatibility with std::vector
	size_t size() const { return buffer_->size(); }
	T* begin() { return &(*buffer_)[0]; }
	T* end() { return &(*buffer_)[buffer_->size() - 1]; }
	T& operator[](const size_t i) { return (*buffer_)[i]; }
	T* data() { return buffer_->data(); }
	const T* data() const { return buffer_->data(); }

private:
	std::shared_ptr<std::vector<T>> buffer_;
};

#endif
// =================================================================================================
#if 1
// Enumeration of buffer access types
enum class BufferAccess { kReadOnly, kWriteOnly, kReadWrite, kNotOwned };

// Tart has this all built-in (but we still need to go through this silliness
// C++11 version of 'cl_mem'
template <typename T>
class Buffer {

public:

	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
	explicit Buffer(const tart::buffer_ptr buffer) : access_(BufferAccess::kNotOwned) { buffer_ = buffer; }

	// Regular constructor with memory management. If this class does not own the buffer object, then
	// the memory will not be freed automatically afterwards. If the size is set to 0, this will
	// become a stub containing a nullptr
	explicit Buffer(const Context& context, const BufferAccess access, const size_t size) :
				access_(access)
	{
		if (size == 0)
			buffer_ = nullptr;
		else
			buffer_ = context.pointer()->allocateBuffer(size*sizeof(T));
	}

	// As above, but now with read/write access as a default
	explicit Buffer(const Context& context, const size_t size) : Buffer<T>(context, BufferAccess::kReadWrite, size) {}

	// Constructs a new buffer based on an existing host-container
	template <typename Iterator>
	explicit Buffer(const Context& context, const Queue& queue, Iterator start, Iterator end)
			: Buffer(context, BufferAccess::kReadWrite, static_cast<size_t>(end - start))
	{
		auto size = static_cast<size_t>(end - start);
		auto pointer = &*start;
		// may be unsafe, but literally the only option here :c
		context.pointer()->allocateBuffer(pointer, size*sizeof(T));
	}

	// Copies from device to host: reading the device buffer a-synchronously
	// (this is currently impossible in tart, so it will just sync for now)
	void ReadAsync(const Queue& queue, const size_t size, T* host, const size_t offset = 0) const
	{
		if (access_ == BufferAccess::kWriteOnly)
		{
			throw LogicError("Buffer: reading from a write-only buffer");
		}
		
		if (offset > 0) throw LogicError("not implemented");
		buffer_->copyOut(host, size*sizeof(T), offset*sizeof(T));
	}
	void ReadAsync(const Queue& queue, const size_t size, std::vector<T>& host, const size_t offset = 0) const
	{
		if (host.size() < size) {
			throw LogicError("Buffer: target host buffer is too small");
		}
		ReadAsync(queue, size, host.data(), offset);
	}
	void ReadAsync(const Queue& queue, const size_t size, BufferHost<T>& host, const size_t offset = 0) const {
		if (host.size() < size) {
			throw LogicError("Buffer: target host buffer is too small");
		}
		ReadAsync(queue, size, host.data(), offset);
	}

	// Copies from device to host: reading the device buffer
	void Read(const Queue& queue, const size_t size, T* host, const size_t offset = 0) const {
		ReadAsync(queue, size, host, offset);
		queue.Finish();
	}
	void Read(const Queue& queue, const size_t size, std::vector<T>& host, const size_t offset = 0) const {
		Read(queue, size, host.data(), offset);
	}
	void Read(const Queue& queue, const size_t size, BufferHost<T>& host, const size_t offset = 0) const {
		Read(queue, size, host.data(), offset);
	}

	// Copies from host to device: writing the device buffer a-synchronously
	void WriteAsync(const Queue& queue, const size_t size, const T* host, const size_t offset = 0) {
		if (access_ == BufferAccess::kReadOnly) {
			throw LogicError("Buffer: writing to a read-only buffer");
		}
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
	void WriteAsync(const Queue& queue, const size_t size, const std::vector<T>& host, const size_t offset = 0) {
		WriteAsync(queue, size, host.data(), offset);
	}
	void WriteAsync(const Queue& queue, const size_t size, const BufferHost<T>& host, const size_t offset = 0) {
		WriteAsync(queue, size, host.data(), offset);
	}

	// Copies from host to device: writing the device buffer
	void Write(const Queue& queue, const size_t size, const T* host, const size_t offset = 0) {
		WriteAsync(queue, size, host, offset);
		queue.Finish();
	}
	void Write(const Queue& queue, const size_t size, const std::vector<T>& host, const size_t offset = 0) {
		Write(queue, size, host.data(), offset);
	}
	void Write(const Queue& queue, const size_t size, const BufferHost<T>& host, const size_t offset = 0) {
		Write(queue, size, host.data(), offset);
	}

	// Copies the contents of this buffer into another device buffer
	void CopyToAsync(const Queue& queue, const size_t size, const Buffer<T>& destination,
									 EventPointer event = nullptr) const {
		if (event != nullptr) throw LogicError("copying with events is not implemented yet");
		buffer_->copyTo(destination(), 0, 0, size*sizeof(T));
		//CheckError(clEnqueueCopyBuffer(queue(), *buffer_, destination(), 0, 0, size * sizeof(T), 0, nullptr, event));
	}
	void CopyTo(const Queue& queue, const size_t size, const Buffer<T>& destination) const {
		CopyToAsync(queue, size, destination);
		queue.Finish();
	}

	// Retrieves the actual allocated size in bytes
	size_t GetSize() const {
		return buffer_->getSize();
	}

	// Accessor to the private data-member
	tart::buffer_ptr operator()() const { return buffer_; }
	
	// get elementwise view of underlying buffer
	tart::buffer_ptr view(size_t nElements) const
	{
		return buffer_->view(nElements*sizeof(T));
	}


private:

	tart::buffer_ptr buffer_;
	BufferAccess access_;
};
#endif

// =================================================================================================

#if 1
typedef std::pair<std::string, tart::program_ptr> kernel_t;

// ahh, right. I remember how it works now
// C++11 version of 'kernel_t'
class Kernel {
	std::string mEntryPoint;
	std::shared_ptr<tart::Program> mProgramContainer = nullptr;
	tart::kernel_ptr mKernel = nullptr;
	tart::device_ptr mDevice;
	// arguments
	kernel_t kernel_;
	std::map<size_t, std::vector<uint8_t>> mNonBufferArgs;
	std::map<size_t, tart::buffer_ptr> mBufferArgs;
public:
	// difference between Vulkan and OpenCL as far as local sizes go will influence this outcome greatly...
	explicit Kernel(const kernel_t kernel) { kernel_ = kernel; }
	explicit Kernel(tart::program_ptr prg) { std::string ep = "none"; kernel_ = {ep, prg}; }
	
	// Regular constructor with memory management
	explicit Kernel(const std::shared_ptr<Program> program, const std::string& name)
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
#if 1
	// Sets a kernel argument at the indicated position
	template <typename T>
	void SetArgument(const size_t index, const T& value) {
		mKernel->setArg(index, value);
	}
#else
	void SetArgument(const size_t index, tart::buffer_ptr value) {
		mBufferArgs[index] = value;
	}

	// Sets a kernel argument at the indicated position
	template <typename T>
	void SetArgument(const size_t index, const T& value) {
		// hmmm....how do we do this?
		// It may require better SPIR-V reflection capability...
		// or will it?
		// i got some tricks up me sleeve!
		// we will just cast it to bytes. easy.
		std::vector<uint8_t> value_cast(sizeof(T));
		std::memcpy(value_cast.data(), &value, sizeof(T));
		mNonBufferArgs[index] = value_cast;
	}
#endif

	// Sets all arguments in one go using parameter packs. Note that this overwrites previously set
	// arguments using 'SetArgument' or 'SetArguments'.
	template <typename... Args>
	void SetArguments(Args&... args) {
		SetArgumentsRecursive(0, args...);
	}

	// Retrieves the amount of local memory used per work-group for this kernel
	unsigned long LocalMemUsage(const Device& device) const {
#if 1
		// It doesn't seem that Vulkan has a direct equivalent to this.
		// More investigation will have to be done.
		// It will likely have something to with SPIR-V reflection...
		return 0;
#else
		const auto bytes = sizeof(uint64_t);
		auto query = kernel_t_work_group_info{CL_KERNEL_LOCAL_MEM_SIZE};
		auto result = uint64_t{0};
		CheckError(clGetKernelWorkGroupInfo(*kernel_, device(), query, bytes, &result, nullptr));
		return static_cast<unsigned long>(result);
#endif
	}

	// Retrieves the name of the kernel
	std::string GetFunctionName() const {
		return mEntryPoint;
	}

	// Launches a kernel onto the specified queue
	void Launch(const Queue& queue, const std::vector<size_t>& global, const std::vector<size_t>& local,
							EventPointer event) {
		const std::vector<Event> dummyWaitlist;
		Launch(queue, global, local, event, dummyWaitlist);
	}

	// As above, but with an event waiting list
	void Launch(const Queue& queue, const std::vector<size_t>& global, const std::vector<size_t>& local,
							EventPointer event, const std::vector<Event>& waitForEvents)
	{
		std::vector<tart::command_sequence_ptr> waitlist(waitForEvents.size());

		// TODO: implement event waiting
		queue.Finish();

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
#if 1
		mKernel->enqueue(adjusted_global, local32);
#else
		tart::pipeline_ptr pipeline = mProgramContainer->getPipeline(mEntryPoint, local32);
		
		// parse push constants
		std::vector<uint8_t> push;
		size_t pushConstIdx = 0;
		for (auto& kv : mNonBufferArgs)
		{
			// copy the push constant to the block at the correct offset
			size_t offset = pipeline->getPushConstantOffset(pushConstIdx);
			push.resize(offset + kv.second.size(), 0);
			
			for (size_t i = 0; i < kv.second.size(); i += 1)
			{
				push[i + offset] = kv.second[i];
			}
			pushConstIdx += 1;
		}
		
		// add buffer args
		std::vector<tart::buffer_ptr> bufs;
		for (auto& kv : mBufferArgs)
		{
			bufs.push_back(kv.second);
		}
		
		tart::command_sequence_ptr sequence = mDevice->createSequence();
		sequence->recordPipeline(pipeline, adjusted_global, bufs, push);
		
		// send it!
		mDevice->submitSequence(sequence);
#endif
	}

	// Accessor to the private data-member
	const kernel_t& operator()() const { return kernel_; }

private:
	// Internal implementation for the recursive SetArguments function.
	template <typename T>
	void SetArgumentsRecursive(const size_t index, T& first) {
		SetArgument(index, first);
	}
	template <typename T, typename... Args>
	void SetArgumentsRecursive(const size_t index, T& first, Args&... args) {
		SetArgument(index, first);
		SetArgumentsRecursive(index + 1, args...);
	}
};
#endif

// =================================================================================================
}	// namespace clblast

// CLBLAST_CLPP11_H_
#endif
