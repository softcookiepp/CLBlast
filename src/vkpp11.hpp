
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

// Exception classes
#include "cxpp11_common.hpp"
#include "tart/tart.hpp"

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
	explicit Device(const size_t device_id);

	// Methods to retrieve device information
	// (platform id is always 0)
	std::string Version() const;
	size_t VersionNumber() const;

	std::string Vendor() const;
	std::string Name() const;
	std::string Type() const;
	size_t MaxWorkGroupSize() const;
	std::vector<size_t> MaxWorkItemSizes() const;

	// Not sure if Tart has a public method for querying extensions; might be a good idea to implement this.
	std::string Capabilities() const;
	bool HasExtension(const std::string& extension) const;
	
	// Vulkan has a way to do this, but I have been too lazy to implement it completely in Tart aside from error checking.
	// Will have to do this eventually
	unsigned long MemorySize() const;
	// this can be retrieved from Tart, but may not be public
	unsigned long MaxAllocSize() const;

	// Query for a specific type of device or brand
	bool IsCPU() const;
	bool IsGPU() const;

	// Accessor to the private data-member
	const RawDeviceID operator()() const;
	
};
#endif

// =================================================================================================

// no idea how to handle this, since Tart uses a single queue
// pretty sure I will just end up scrapping it, since tart::Device handles all this already (actually I can't)
// C++11 version of 'cl_command_queue'
class Queue {
	tart::device_ptr mDevice = nullptr;
public:
	// Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
	explicit Queue(const tart::device_ptr queue);

	// Regular constructor with memory management
	explicit Queue(const Device& device);

	// Synchronizes the queue
	void Finish() const;

	// Retrieves the corresponding context or device
	Device GetDevice() const;

	// Accessor to the private data-member
	const tart::device_ptr& operator()() const;
};

// =================================================================================================
#if 1

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
	explicit Buffer(const tart::device_ptr& device, const size_t size);

	// Copies from device to host: reading the device buffer a-synchronously
	// (this is currently impossible in tart, so it will just sync for now)
	void ReadAsync(const Queue& queue, const size_t size, T* host, const size_t offset = 0) const;
	void ReadAsync(const Queue& queue, const size_t size, std::vector<T>& host, const size_t offset = 0) const;

	// Copies from device to host: reading the device buffer
	void Read(const Queue& queue, const size_t size, T* host, const size_t offset = 0) const;
	void Read(const Queue& queue, const size_t size, std::vector<T>& host, const size_t offset = 0) const;

	// Copies from host to device: writing the device buffer a-synchronously
	void WriteAsync(const Queue& queue, const size_t size, const T* host, const size_t offset = 0);
	void WriteAsync(const Queue& queue, const size_t size, const std::vector<T>& host, const size_t offset = 0);

	// Copies from host to device: writing the device buffer
	void Write(const Queue& queue, const size_t size, const T* host, const size_t offset = 0);
	void Write(const Queue& queue, const size_t size, const std::vector<T>& host, const size_t offset = 0);

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
};
#endif

// =================================================================================================

// Kernel wrapper: provides OpenCL-like SetArgument() API over tart::kernel_ptr
class Kernel {
	tart::kernel_ptr mKernel;
public:
	explicit Kernel(const tart::kernel_ptr kernel) : mKernel(kernel) {}

	// Sets push constant at index
	template <typename T>
	void SetArgument(const size_t index, const T& value) {
		mKernel->setArg(index, value);
	}

	// Expose underlying tart::kernel_ptr for TimeKernel and setArgAs
	tart::kernel_ptr get() const { return mKernel; }

	// Accessor
	const tart::kernel_ptr& operator()() const { return mKernel; }
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_CLPP11_H_
#endif
