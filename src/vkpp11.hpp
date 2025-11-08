
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
#define cl_device_info size_t
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

typedef std::vector<tart::command_sequence_ptr> sequence_ptr_list;

// not used in tart (yet)
// however, for API compatibility, it needs to be preserved
// C++11 version of 'cl_event'
class Event {
	// An event, in regards to Tart compatibility (for now)
	// will just be a list of sequences that the device will away the completion of.
	// which means Tart must be modified to be able to get the parent device from a sequence?
	// ugh
	std::shared_ptr<sequence_ptr_list> mSequences = nullptr;
public:

	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
	explicit Event(const sequence_ptr_list& event) { mSequences = std::make_shared<sequence_ptr_list>(event); }

	// Regular constructor with memory management
	explicit Event()
	{
		mSequences = std::make_shared<std::vector<tart::command_sequence_ptr> >();
	}

	// Waits for completion of this event
	void WaitForCompletion() const
	{
		if (mSequences->size() > 0)
		{
			// devices should (hopefully) all be the same across sequences
			tart::device_ptr device = (*mSequences)[0]->getDevice();
			device->sync(*mSequences);
		}
	}

	// Retrieves the elapsed time of the last recorded event.
	// (Note that there is a bug in Apple's OpenCL implementation of the 'clGetEventProfilingInfo' function:
	//	http://stackoverflow.com/questions/26145603/clgeteventprofilinginfo-bug-in-macosx)
	// However, in our case the reply size is fixed to be cl_ulong, so we are not affected.
	float GetElapsedTime() const {
#if 1
		// not implemented yet :c
		return 1.0;
#else
		WaitForCompletion();
		const auto bytes = sizeof(cl_ulong);
		auto time_start = cl_ulong{0};
		CheckError(clGetEventProfilingInfo(*event_, CL_PROFILING_COMMAND_START, bytes, &time_start, nullptr));
		auto time_end = cl_ulong{0};
		CheckError(clGetEventProfilingInfo(*event_, CL_PROFILING_COMMAND_END, bytes, &time_end, nullptr));
		return static_cast<float>(time_end - time_start) * 1.0e-6f;
#endif
	}

	// Accessor to the private data-member
	sequence_ptr_list& operator()() { return *mSequences; }
	const sequence_ptr_list& operator()() const { return *mSequences; }
	std::shared_ptr<sequence_ptr_list> pointer() { return mSequences; }
	const std::shared_ptr<sequence_ptr_list> pointer() const { return mSequences; }
};


// Pointer to...this dumb crap
using EventPointer = std::shared_ptr<sequence_ptr_list>;

// =================================================================================================

// Raw platform ID type
// since Vulkan doesn't have platforms like OpenCL does, there will only be a single platform that has all the devices.
using RawPlatformID = size_t;

// Vulkan doesn't have any direct equivalent to this, just use it to encapsulate device..or maybe instance?
class Platform {
	// we need this to get the number of devices
	tart::Instance mDummyInstance;
public:
	// Initializes the platform
	explicit Platform(const size_t platform_id) {
		// there can only be one, this is Vulkan c:
		if (platform_id != 0) {
		throw LogicError("Vulkan back-end requires a platform ID of 0");
	}

	// Methods to retrieve platform information
	std::string Name() const {
#if 1
		return "not implemented";
#else
		return GetInfoString(CL_PLATFORM_NAME);
#endif
	}
	std::string Vendor() const { return GetInfoString(CL_PLATFORM_VENDOR); }
	std::string Version() const { return GetInfoString(CL_PLATFORM_VERSION); }

	// Returns the number of devices on this platform
	size_t NumDevices() const {
		return static_cast<size_t>mDummyInstance.getNumDevices();
	}

	// Accessor to the private data-member
	const RawPlatformID& operator()() const { return platform_; }

private:
	cl_platform_id platform_;

	// Private helper functions
	std::string GetInfoString(const cl_device_info info) const {
#if 1
		// no idea what this is supposed to do; we find out L A T E R
		return "not implemented";
#else
		auto bytes = size_t{0};
		CheckError(clGetPlatformInfo(platform_, info, 0, nullptr, &bytes));
		auto result = std::string{};
		result.resize(bytes);
		CheckError(clGetPlatformInfo(platform_, info, bytes, &result[0], nullptr));
		result.resize(strlen(result.c_str()));	// Removes any trailing '\0'-characters
		return result;
#endif
	}
};

// not applicable for vulkan
// Retrieves a vector with all platforms
inline std::vector<Platform> GetAllPlatforms() {
#if 1
	// TODO: populate this based on single vulkan devices.
	std::vector<Platform> platforms({{0}});
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
		raise RuntimeError("not implemented! (unsure of how to handle multiple tart::Instance...");
	}

	// Methods to retrieve device information
	// (platform id is always 0)
	RawPlatformID PlatformID() const { return 0; }
	std::string Version() const { return "Vulkan 1.2"; } // pretty sure this will work?
	size_t VersionNumber() const {
		return 120;
	}
	// TODO: implement some of this stuff in tart
	std::string Vendor() const { return "vendor name not implemented"; }
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
		return static_cast<unsigned long>(GetInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE));
#endif
	}

	// Not sure if Tart has a public method for querying extensions; might be a good idea to implement this.
	std::string Capabilities() const { return GetInfoString(CL_DEVICE_EXTENSIONS); }
	bool HasExtension(const std::string& extension) const {
		return mDevice->supportsExtension(extension);
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
	unsigned long MemorySize() const { return static_cast<unsigned long>(GetInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE)); }
	// this can be retrieved from Tart, but may not be public
	unsigned long MaxAllocSize() const {
		return static_cast<unsigned long>(GetInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE));
	}
	
	// neither of these can be queried in Vulkan either
	size_t MemoryClock() const { return 0; }		 // Not exposed in OpenCL
	size_t MemoryBusWidth() const { return 0; }	// Not exposed in OpenCL

	// Configuration-validity checks
	bool IsLocalMemoryValid(const cl_ulong local_mem_usage) const
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
#if 1
		// Not implemented yet. Will have to figure this out later :c
		return false;
#else
		return Vendor() == "AMD" || Vendor() == "Advanced Micro Devices, Inc." || Vendor() == "AuthenticAMD";
#endif
	}
	bool IsNVIDIA() const
	{
#if 1
		// just assume true for now, since testing devices are of the leather jacket variety
		return true;
#else
		return Vendor() == "NVIDIA" || Vendor() == "NVIDIA Corporation";
#endif
	}
	bool IsIntel() const
	{
#if 1
		// nope
		return false;
#else
		return Vendor() == "INTEL" || Vendor() == "Intel" || Vendor() == "GenuineIntel" ||
					 Vendor() == "Intel(R) Corporation";
#endif
	}
	bool IsARM() const
	{
#if 1
		return false;
#else
		return Vendor() == "ARM";
#endif
	}
	bool IsQualcomm() const
	{
#if 1
		return false;
#else
		return Vendor() == "QUALCOMM";
#endif
	}

	// Platform specific extensions
	std::string AMDBoardName() const
	{
#if 1
		return "not implemented";
#else
		// check for 'cl_amd_device_attribute_query' first
#ifndef CL_DEVICE_BOARD_NAME_AMD
#define CL_DEVICE_BOARD_NAME_AMD 0x4038
#endif
		return GetInfoString(CL_DEVICE_BOARD_NAME_AMD);
	}
	std::string NVIDIAComputeCapability() const {	// check for 'cl_nv_device_attribute_query' first
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
#endif
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
#endif
		return std::string{"SM"} + std::to_string(GetInfo<cl_uint>(CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV)) +
					 std::string{"."} + std::to_string(GetInfo<cl_uint>(CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV));
#endif
	}

	// Returns if the Nvidia chip is a Volta or later archicture (sm_70 or higher)
	bool IsPostNVIDIAVolta() const
	{
#if 1
		return false;
#else
		if (HasExtension("cl_nv_device_attribute_query")) {
			return GetInfo<cl_uint>(CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV) >= 7;
		}
		return false;
#endif
	}

	// Returns the Qualcomm Adreno GPU version (i.e. a650, a730, a740, etc.)
	std::string AdrenoVersion() const
	{
#if 1
		return "not implemented";
#else
		if (IsQualcomm()) {
			return GetInfoString(CL_DEVICE_OPENCL_C_VERSION);
		} else {
			return std::string{""};
		}
#endif
	}

	// Retrieves the above extra information (if present)
	std::string GetExtraInfo() const
	{
#if 1
		return "not implemented";
#else
		if (HasExtension("cl_amd_device_attribute_query")) {
			return AMDBoardName();
		}
		if (HasExtension("cl_nv_device_attribute_query")) {
			return NVIDIAComputeCapability();
		} else {
			return std::string{""};
		}
#endif
	}

	// Accessor to the private data-member
	const RawDeviceID operator()() const { return mDevice; }
	
private:

	// Private helper functions
	template <typename T>
	T GetInfo(const cl_device_info info) const {
		auto bytes = size_t{0};
		CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
		auto result = T(0);
		CheckError(clGetDeviceInfo(device_, info, bytes, &result, nullptr));
		return result;
	}
	template <typename T>
	std::vector<T> GetInfoVector(const cl_device_info info) const {
		auto bytes = size_t{0};
		CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
		auto result = std::vector<T>(bytes / sizeof(T));
		CheckError(clGetDeviceInfo(device_, info, bytes, result.data(), nullptr));
		return result;
	}
	std::string GetInfoString(const cl_device_info info) const {
		auto bytes = size_t{0};
		CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
		auto result = std::string{};
		result.resize(bytes);
		CheckError(clGetDeviceInfo(device_, info, bytes, &result[0], nullptr));
		result.resize(strlen(result.c_str()));	// Removes any trailing '\0'-characters
		return result;
	}
};
// =================================================================================================

// ok, so the `Device` is more like `vk::PhysicalDevice` and the `Context` is more akin to `vk::Device`
// that makes sense.

// Raw context type
using RawContext = tart::device_ptr;

// C++11 version of 'cl_context'
class Context {

public:
	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
	explicit Context(const cl_context context) : context_(new cl_context) { *context_ = context; }

	// Regular constructor with memory management
	explicit Context(const Device& device)
			: context_(new cl_context, [](cl_context* c) {
					if (*c) {
						CheckErrorDtor(clReleaseContext(*c));
					}
					delete c;
				}) {
		auto status = CL_SUCCESS;
		const cl_device_id dev = device();
		*context_ = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
		CLCudaAPIError::Check(status, "clCreateContext");
	}

	// Accessor to the private data-member
	const RawContext& operator()() const { return *context_; }
	RawContext* pointer() const { return &(*context_); }

private:
	std::shared_ptr<cl_context> context_;
};

// Pointer to an OpenCL context
using ContextPointer = cl_context*;
// =================================================================================================

// C++11 version of 'cl_program'.
// I may keep this abstraction, actually
class Program {
	std::string mSource;
	tart::device_ptr mDevice = nullptr;
	tart::shader_module_ptr mShaderModule = nullptr;
	tart::cl_program_ptr mCLProgram = nullptr;
public:
	// Source-based constructor with memory management
	explicit Program(tart::device_ptr device, const std::string& source) {
		mSource = source;
		mDevice = device;
	}

	// Binary-based constructor with memory management
	explicit Program(tart::device_ptr device, std::vector<uint32_t>& binary) {
		mDevice = device;
		mShaderModule = device->loadShader(binary);
		mCLProgram = device->createCLProgram(mShaderModule);
	}

	// Compiles the device program and checks whether or not there are any warnings/errors
	void Build(std::vector<std::string>& options) {
		// TODO: parse options (look for dflags, etc.)
		// compile the shader module
		mShaderModule = mDevice->compileCL(mSource);
		// load it into the actual CL program handler thingy
		mCLProgram = mDevice->createCLProgram();
#if 0
		// this is where the options happen.
		auto options_string = std::accumulate(options.begin(), options.end(), std::string{" "});
#endif
	}

	// Confirms whether a certain status code is an actual compilation error or warning
	bool StatusIsCompilationWarningOrError(const int32_t status) const { return (status == -11); }

	// Retrieves the warning/error message from the compiler (if any)
	std::string GetBuildInfo() const {
		return "not implemented yet :c";
	}

	// Retrieves a binary or an intermediate representation of the compiled program
	std::string GetIR() const {
		// TODO: somehow convert the SPIR-V into a string?
		return "not implemented yet :c";
	}

	// Accessor to the private data-member
	tart::cl_program_ptr operator()() const { return mCLProgram; }
};

// =================================================================================================
// Tart does not expose queues (yet?)
// since each device only has a single compute queue
#if 0
// Raw command-queue type
using RawCommandQueue = cl_command_queue;

// no idea how to handle this, since Tart uses a single queue
// pretty sure I will just end up scrapping it, since tart::Device handles all this already
// C++11 version of 'cl_command_queue'
class Queue {
public:
	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
	explicit Queue(const cl_command_queue queue) : queue_(new cl_command_queue) { *queue_ = queue; }

	// Regular constructor with memory management
	explicit Queue(const Context& context, const Device& device)
			: queue_(new cl_command_queue, [](cl_command_queue* s) {
					if (*s) {
						CheckErrorDtor(clReleaseCommandQueue(*s));
					}
					delete s;
				}) {
		auto status = CL_SUCCESS;
		*queue_ = clCreateCommandQueue(context(), device(), CL_QUEUE_PROFILING_ENABLE, &status);
		CLCudaAPIError::Check(status, "clCreateCommandQueue");
	}

	// Synchronizes the queue
	void Finish(Event&) const { Finish(); }
	void Finish() const { CheckError(clFinish(*queue_)); }

	// Retrieves the corresponding context or device
	Context GetContext() const {
		auto bytes = size_t{0};
		CheckError(clGetCommandQueueInfo(*queue_, CL_QUEUE_CONTEXT, 0, nullptr, &bytes));
		cl_context result;
		CheckError(clGetCommandQueueInfo(*queue_, CL_QUEUE_CONTEXT, bytes, &result, nullptr));
		return Context(result);
	}
	Device GetDevice() const {
		auto bytes = size_t{0};
		CheckError(clGetCommandQueueInfo(*queue_, CL_QUEUE_DEVICE, 0, nullptr, &bytes));
		cl_device_id result;
		CheckError(clGetCommandQueueInfo(*queue_, CL_QUEUE_DEVICE, bytes, &result, nullptr));
		return Device(result);
	}

	// Accessor to the private data-member
	const RawCommandQueue& operator()() const { return *queue_; }

private:
	std::shared_ptr<cl_command_queue> queue_;
};
#endif
// =================================================================================================

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

// =================================================================================================

#if 1
// Enumeration of buffer access types
enum class BufferAccess { kReadOnly, kWriteOnly, kReadWrite, kNotOwned };

// Tart has this all built-in.
// C++11 version of 'cl_mem'
template <typename T>
class Buffer {

public:

	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
	explicit Buffer(const cl_mem buffer) : buffer_(new cl_mem), access_(BufferAccess::kNotOwned) { *buffer_ = buffer; }

	// Regular constructor with memory management. If this class does not own the buffer object, then
	// the memory will not be freed automatically afterwards. If the size is set to 0, this will
	// become a stub containing a nullptr
	explicit Buffer(const Context& context, const BufferAccess access, const size_t size)
			: buffer_(new cl_mem,
								[access, size](cl_mem* m) {
									if (access != BufferAccess::kNotOwned && size > 0) {
										CheckError(clReleaseMemObject(*m));
									}
									delete m;
								}),
				access_(access) {
		auto flags = cl_mem_flags{CL_MEM_READ_WRITE};
		if (access_ == BufferAccess::kReadOnly) {
			flags = CL_MEM_READ_ONLY;
		}
		if (access_ == BufferAccess::kWriteOnly) {
			flags = CL_MEM_WRITE_ONLY;
		}
		auto status = CL_SUCCESS;
		*buffer_ = (size > 0) ? clCreateBuffer(context(), flags, size * sizeof(T), nullptr, &status) : nullptr;
		CLCudaAPIError::Check(status, "clCreateBuffer");
	}

	// As above, but now with read/write access as a default
	explicit Buffer(const Context& context, const size_t size) : Buffer<T>(context, BufferAccess::kReadWrite, size) {}

	// Constructs a new buffer based on an existing host-container
	template <typename Iterator>
	explicit Buffer(const Context& context, const Queue& queue, Iterator start, Iterator end)
			: Buffer(context, BufferAccess::kReadWrite, static_cast<size_t>(end - start)) {
		auto size = static_cast<size_t>(end - start);
		auto pointer = &*start;
		CheckError(clEnqueueWriteBuffer(queue(), *buffer_, CL_FALSE, 0, size * sizeof(T), pointer, 0, nullptr, nullptr));
		queue.Finish();
	}

	// Copies from device to host: reading the device buffer a-synchronously
	void ReadAsync(const Queue& queue, const size_t size, T* host, const size_t offset = 0) const {
		if (access_ == BufferAccess::kWriteOnly) {
			throw LogicError("Buffer: reading from a write-only buffer");
		}
		CheckError(clEnqueueReadBuffer(queue(), *buffer_, CL_FALSE, offset * sizeof(T), size * sizeof(T), host, 0, nullptr,
																	 nullptr));
	}
	void ReadAsync(const Queue& queue, const size_t size, std::vector<T>& host, const size_t offset = 0) const {
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
		CheckError(clEnqueueWriteBuffer(queue(), *buffer_, CL_FALSE, offset * sizeof(T), size * sizeof(T), host, 0, nullptr,
																		nullptr));
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
		CheckError(clEnqueueCopyBuffer(queue(), *buffer_, destination(), 0, 0, size * sizeof(T), 0, nullptr, event));
	}
	void CopyTo(const Queue& queue, const size_t size, const Buffer<T>& destination) const {
		CopyToAsync(queue, size, destination);
		queue.Finish();
	}

	// Retrieves the actual allocated size in bytes
	size_t GetSize() const {
		const auto bytes = sizeof(size_t);
		auto result = size_t{0};
		CheckError(clGetMemObjectInfo(*buffer_, CL_MEM_SIZE, bytes, &result, nullptr));
		return result;
	}

	// Accessor to the private data-member
	const cl_mem& operator()() const { return *buffer_; }


private:

	std::shared_ptr<cl_mem> buffer_;
	BufferAccess access_;
};
#endif
// =================================================================================================

// ahh, right. I remember how it works now
// C++11 version of 'cl_kernel'
class Kernel {
	std::string mEntryPoint;
	tart::cl_program_ptr mCLProgram;
	// arguments
	std::map<size_t, std::vector<uint8_t> mNonBufferArgs;
	std::map<size_t, tart::buffer_ptr> mBufferArgs;
public:
#if 0
	// disabling until I know how to implement it.
	// difference between Vulkan and OpenCL as far as local sizes go will influence this outcome greatly...
	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
	explicit Kernel(const cl_kernel kernel) : kernel_(new cl_kernel) { *kernel_ = kernel; }
#endif
	// Regular constructor with memory management
	explicit Kernel(const std::shared_ptr<Program> program, const std::string& name)
	{
		// this will be a bit different.
		// OpenCL allows kernels to be created that accept a variable local size.
		// Vulkan allows pipelines to be created where the entry point is specified,
		// but a fixed local size is used.
		// tart::CLProgram takes care of this, but it must be adapted to this library
		mEntryPoint = name;
		mCLProgram = program->operator()();
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
		std::memcpy(value_cast.data, &value, sizeof(T));
		mNonBufferArgs[index] = value_cast;
	}
	template <typename T>
	void SetArgument(const size_t index, tart::buffer_ptr value) {
		mBufferArgs[index] = value;
	}

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
		const auto bytes = sizeof(cl_ulong);
		auto query = cl_kernel_work_group_info{CL_KERNEL_LOCAL_MEM_SIZE};
		auto result = cl_ulong{0};
		CheckError(clGetKernelWorkGroupInfo(*kernel_, device(), query, bytes, &result, nullptr));
		return static_cast<unsigned long>(result);
#endif
	}

	// Retrieves the name of the kernel
	std::string GetFunctionName() const {
#if 1
		return mEntryPoint;
#else
		auto bytes = size_t{0};
		CheckError(clGetKernelInfo(*kernel_, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &bytes));
		auto result = std::string{};
		result.resize(bytes);
		CheckError(clGetKernelInfo(*kernel_, CL_KERNEL_FUNCTION_NAME, bytes, &result[0], nullptr));
		return std::string{result.c_str()};	// Removes any trailing '\0'-characters
#endif
	}

	// Launches a kernel onto the specified queue
	void Launch(const Queue& queue, const std::vector<size_t>& global, const std::vector<size_t>& local,
							EventPointer event) {
#if 1
		
#else
		CheckError(clEnqueueNDRangeKernel(queue(), *kernel_, static_cast<cl_uint>(global.size()), nullptr, global.data(),
																			local.data(), 0, nullptr, event));
#endif
	}

	// As above, but with an event waiting list
	void Launch(const Queue& queue, const std::vector<size_t>& global, const std::vector<size_t>& local,
							EventPointer event, const std::vector<Event>& waitForEvents) {
		// Builds a plain version of the events waiting list
		auto waitForEventsPlain = std::vector<cl_event>();
		for (auto& waitEvent : waitForEvents) {
			if (waitEvent()) {
				waitForEventsPlain.push_back(waitEvent());
			}
		}

		// Launches the kernel while waiting for other events
		CheckError(clEnqueueNDRangeKernel(queue(), *kernel_, static_cast<cl_uint>(global.size()), nullptr, global.data(),
																			!local.empty() ? local.data() : nullptr,
																			static_cast<cl_uint>(waitForEventsPlain.size()),
																			!waitForEventsPlain.empty() ? waitForEventsPlain.data() : nullptr, event));
	}

	// Accessor to the private data-member
	const cl_kernel& operator()() const { return *kernel_; }

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

// =================================================================================================
}	// namespace clblast

// CLBLAST_CLPP11_H_
#endif
