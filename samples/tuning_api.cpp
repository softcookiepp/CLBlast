
// =================================================================================================
// This file demonstrates the use of the runtime tuning API with CLBlast-vk.
// It is a stand-alone example that uses the Vulkan/Tart backend.
// =================================================================================================

#include <chrono>
#include <cstdio>
#include <vector>

// Includes the CLBlast-vk library
#include <clblast_vk.h>

// =================================================================================================

int main() {
  // Vulkan device settings
  const auto device_id = 0;

  // Example arguments
  const size_t m = 128;
  const size_t n = 64;
  const auto fraction = 1.0;  // between 0.0 and 1.0

  // Initialize Vulkan device via Tart
  tart::Instance instance;
  if (instance.getDeviceCount() == 0 || device_id >= instance.getDeviceCount()) {
    fprintf(stderr, "No Vulkan device available\n");
    return 1;
  }
  auto device = instance.getDevice(device_id);

  // Creates a Tart command queue
  auto queue = device->createQueue();

  // Performs the tuning
  printf("Starting the tuning...\n");
  std::unordered_map<std::string, size_t> parameters;
  auto tart_queue = queue.get();
  auto status = clblast::TuneCopy<float>(&tart_queue, m, n, fraction, parameters);

  // Tuning completed. See "clblast_vk.h" for status codes (0 -> success).
  printf("Completed TuneCopy with status %d (0 == OK), found parameters:\n", static_cast<int>(status));
  for (const auto& parameter : parameters) {
    printf(">  %s = %zu\n", parameter.first.c_str(), parameter.second);
  }

  // Set the new parameters
  status = clblast::OverrideParameters(device(), "Copy", clblast::Precision::kSingle, parameters);
  printf("Completed OverrideParameters with status %d (0 == OK)\n", static_cast<int>(status));
  return 0;
}

// =================================================================================================
