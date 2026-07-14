
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
#include <iostream>
#include <complex> // std::complex<float>, std::complex<double>


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
	explicit CLCudaAPIError(int32_t status, const std::string& where);
	static void Check(const int32_t status, const std::string& where);

	static void CheckDtor(const int32_t status, const std::string& where);
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
	explicit Event(const tart::event_ptr event);

	// Regular constructor with memory management
	explicit Event();

	// Waits for completion of this event
	void WaitForCompletion() const;

	// Retrieves the elapsed time of the last recorded event.
	// (Note that there is a bug in Apple's OpenCL implementation of the 'clGetEventProfilingInfo' function:
	//	http://stackoverflow.com/questions/26145603/clgeteventprofilinginfo-bug-in-macosx)
	// However, in our case the reply size is fixed to be uint64_t, so we are not affected.
	float GetElapsedTime() const;

	// Accessor to the private data-member
	tart::event_ptr operator()();
	const tart::event_ptr operator()() const;
	tart::event_ptr  pointer();
	const tart::event_ptr pointer() const;
};

#if 1
// =================================================================================================

// Vulkan doesn't have any direct equivalent to this, just use it to encapsulate device..or maybe instance?
class Platform {
	std::shared_ptr<tart::Instance> mInstance = nullptr;
public:
	// Initializes the platform
	explicit Platform(const size_t platform_id);

	// Methods to retrieve platform information
	std::string Name() const;
	std::string Vendor() const;
	std::string Version() const;
	
	// returns the tart::Instance
	tart::Instance& getInstance() const;

	// Returns the number of devices on this platform
	size_t NumDevices();

	// Accessor to the private data-member
	const size_t& operator()() const;

private:
	size_t platform_ = 0;
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
	explicit Device(const tart::device_ptr device);

	// Initialize the device. Note that this constructor can throw exceptions!
	explicit Device(const Platform& platform, const size_t device_id);

	// Methods to retrieve device information
	// (platform id is always 0)
	size_t PlatformID() const;
	std::string Version() const;
	size_t VersionNumber() const;

	std::string Vendor() const;
	std::string Name() const;
	std::string Type() const;
	size_t MaxWorkGroupSize() const;
	size_t MaxWorkItemDimensions() const;
	std::vector<size_t> MaxWorkItemSizes() const;
	unsigned long LocalMemSize() const;

	// Not sure if Tart has a public method for querying extensions; might be a good idea to implement this.
	std::string Capabilities() const;
	bool HasExtension(const std::string& extension) const;
	
	// Tart already has this
	bool SupportsFP64() const;
	bool SupportsFP16() const;
	// Vulkan does not allow you to do this
	size_t CoreClock() const;
	// or this either.
	size_t ComputeUnits() const;
	
	// Vulkan has a way to do this, but I have been too lazy to implement it completely in Tart aside from error checking.
	// Will have to do this eventually
	unsigned long MemorySize() const;
	// this can be retrieved from Tart, but may not be public
	unsigned long MaxAllocSize() const;
	
	// neither of these can be queried in Vulkan either
	size_t MemoryClock() const;
	size_t MemoryBusWidth() const;

	// Configuration-validity checks
	bool IsLocalMemoryValid(const uint64_t local_mem_usage) const;
	bool IsThreadConfigValid(const std::vector<size_t>& local) const;

	// Query for a specific type of device or brand
	bool IsCPU() const;
	bool IsGPU() const;
	bool IsAMD() const;
	bool IsNVIDIA() const;
	bool IsIntel() const;
	bool IsARM() const;
	bool IsQualcomm() const;

	// Platform specific extensions
	std::string AMDBoardName() const;
	std::string NVIDIAComputeCapability() const;

	// Returns if the Nvidia chip is a Volta or later archicture (sm_70 or higher)
	bool IsPostNVIDIAVolta() const;

	// Returns the Qualcomm Adreno GPU version (i.e. a650, a730, a740, etc.)
	std::string AdrenoVersion() const;

	// Retrieves the above extra information (if present)
	std::string GetExtraInfo() const;

	// Accessor to the private data-member
	const RawDeviceID operator()() const;
	
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
	explicit Context(tart::device_ptr context);

	// Regular constructor with memory management
	explicit Context(const Device& device);

	// Accessor to the private data-member
	const RawContext operator()() const;
	RawContext pointer() const;
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
	explicit Program(const clblast::Context& context, const std::string& source);
	
	// constructor for GLSL shaders
	// requires multiple shader sources because each file can only have one entry point :c
	explicit Program(const clblast::Context& context, std::map<std::string, std::string>& kernelSources);

	// Binary-based constructor with memory management
	explicit Program(const clblast::Device& device, const clblast::Context& context, std::string& binary);

	// Compiles the device program and checks whether or not there are any warnings/errors
	void Build(const clblast::Device& device, const clblast::Context& context, std::vector<std::string>& options);
	
	// Compiles the device program and checks whether or not there are any warnings/errors
	void Build(const clblast::Device& device, std::vector<std::string>& options);

	// Confirms whether a certain status code is an actual compilation error or warning
	bool StatusIsCompilationWarningOrError(const int32_t status) const;

	// Retrieves the warning/error message from the compiler (if any)
	std::string GetBuildInfo(const clblast::Device& device) const;

	// Retrieves a binary or an intermediate representation of the compiled program
	std::string GetIR() const;

	// Accessor to the private data-member
	std::shared_ptr<tart::Program> operator()() const;
};
#endif

// =================================================================================================
// Tart does not expose queues (yet?)
// since each device only has a single compute queue
// Raw command-queue type
using RawCommandQueue = tart::device_ptr;

// no idea how to handle this, since Tart uses a single queue
// pretty sure I will just end up scrapping it, since tart::Device handles all this already (actually I can't)
// C++11 version of 'cl_command_queue'
class Queue {
	tart::device_ptr mDevice = nullptr;
public:
	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
	explicit Queue(const tart::device_ptr queue);

	// Regular constructor with memory management
	explicit Queue(const Context& context, const Device& device);

	// Synchronizes the queue
	void Finish(Event& event) const;
	void Finish() const;

	// Retrieves the corresponding context or device
	Context GetContext() const;
	Device GetDevice() const;

	// Accessor to the private data-member
	const RawCommandQueue& operator()() const;
};

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
	explicit Buffer(const tart::buffer_ptr buffer);

	// Regular constructor with memory management. If this class does not own the buffer object, then
	// the memory will not be freed automatically afterwards. If the size is set to 0, this will
	// become a stub containing a nullptr
	explicit Buffer(const Context& context, const BufferAccess access, const size_t size);

	// As above, but now with read/write access as a default
	explicit Buffer(const Context& context, const size_t size);

	// Constructs a new buffer based on an existing host-container
	// Keep this in the header, since keeping track of what Iterator is used will be a pain
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
	void ReadAsync(const Queue& queue, const size_t size, T* host, const size_t offset = 0) const;
	void ReadAsync(const Queue& queue, const size_t size, std::vector<T>& host, const size_t offset = 0) const;
	void ReadAsync(const Queue& queue, const size_t size, BufferHost<T>& host, const size_t offset = 0) const;

	// Copies from device to host: reading the device buffer
	void Read(const Queue& queue, const size_t size, T* host, const size_t offset = 0) const;
	void Read(const Queue& queue, const size_t size, std::vector<T>& host, const size_t offset = 0) const;
	void Read(const Queue& queue, const size_t size, BufferHost<T>& host, const size_t offset = 0) const;

	// Copies from host to device: writing the device buffer a-synchronously
	void WriteAsync(const Queue& queue, const size_t size, const T* host, const size_t offset = 0);
	void WriteAsync(const Queue& queue, const size_t size, const std::vector<T>& host, const size_t offset = 0);
	void WriteAsync(const Queue& queue, const size_t size, const BufferHost<T>& host, const size_t offset = 0);

	// Copies from host to device: writing the device buffer
	void Write(const Queue& queue, const size_t size, const T* host, const size_t offset = 0);
	void Write(const Queue& queue, const size_t size, const std::vector<T>& host, const size_t offset = 0);
	void Write(const Queue& queue, const size_t size, const BufferHost<T>& host, const size_t offset = 0);

	// Copies the contents of this buffer into another device buffer
	void CopyToAsync(const Queue& queue, const size_t size, const Buffer<T>& destination,
									 EventPointer event = nullptr) const;
	void CopyTo(const Queue& queue, const size_t size, const Buffer<T>& destination) const ;

	// Retrieves the actual allocated size in bytes
	size_t GetSize() const;

	// Accessor to the private data-member
	tart::buffer_ptr operator()() const;


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
public:
	// difference between Vulkan and OpenCL as far as local sizes go will influence this outcome greatly...
	explicit Kernel(const kernel_t kernel);
	explicit Kernel(tart::program_ptr prg);
	
	// Regular constructor with memory management
	explicit Kernel(const std::shared_ptr<Program> program, const std::string& name);

	// Sets a kernel argument at the indicated position
	template <typename T>
	void SetArgument(const size_t index, const T& value) {
		mKernel->setArg(index, value);
	}

	// Sets all arguments in one go using parameter packs. Note that this overwrites previously set
	// arguments using 'SetArgument' or 'SetArguments'.
	template <typename... Args>
	void SetArguments(Args&... args) {
		SetArgumentsRecursive(0, args...);
	}

	// Retrieves the amount of local memory used per work-group for this kernel
	unsigned long LocalMemUsage(const Device& device) const;

	// Retrieves the name of the kernel
	std::string GetFunctionName() const;

	// As above, but with an event waiting list
	void Launch(const Queue& queue, const std::vector<size_t>& global, const std::vector<size_t>& local,
							EventPointer event, const std::vector<Event>& waitForEvents = {});

	// Accessor to the private data-member
	const kernel_t& operator()() const;

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
