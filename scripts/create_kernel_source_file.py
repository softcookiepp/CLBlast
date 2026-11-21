import os
import sys
import struct

def read_spv(filepath):
	integer_list = []
	
	FORMAT = '<I'  # Little-endian unsigned 32-bit integer (4 bytes)
	INT_SIZE = struct.calcsize(FORMAT) # This will be 4 bytes
	
	with open(filepath, 'rb') as f:
		while True:
			# 1. Read exactly 4 bytes (the size of one 32-bit integer)
			binary_chunk = f.read(INT_SIZE)
			
			# 2. Check if we have read a full integer
			if not binary_chunk:
				# End of file reached
				break
			
			# 3. Unpack the bytes into an integer
			# struct.unpack returns a tuple, so we take the first element [0]
			value = struct.unpack(FORMAT, binary_chunk)[0]
			integer_list.append(value)
	assert integer_list[0] == 0x07230203 #SPIR-V magic number
	return integer_list

def render_source_file(spv_directory, source_file_name, sources, persistent = False):
	full_hpp = """
#include "tart.hpp"

namespace tart
{
namespace kernels
{
	"""
	full_cpp = """
#include "$KERNEL_SOURCE_FILENAME.hpp"

namespace tart
{
namespace kernels
{

#ifndef LOAD_CACHED_KERNEL
#define LOAD_CACHED_KERNEL(DEVICE, KERNEL_SRC) \
{ \
	static std::map<std::uintptr_t, kernel_ptr> deviceKernels; \
	std::uintptr_t key = (std::uintptr_t)DEVICE.get(); \
	if (deviceKernels.find(key) == deviceKernels.end()) \
	{ \
		const KernelSourceImpl& k = KERNEL_SRC; \
		deviceKernels[key] = DEVICE->loadShader(k.spv, k.name)->createKernel(k.entrypoint); \
	} \
	return deviceKernels[key]; \
}
#endif
	""".replace("$KERNEL_SOURCE_FILENAME", source_file_name)
	for namespace, ns_source_list in sources.items():
		ns_str = """
namespace $NAMESPACE
{

		""".replace("$NAMESPACE", namespace)
		full_cpp += ns_str
		full_hpp += ns_str
		for src in ns_source_list:
			spv_cpp_str = "{" + ", ".join([str(i) for i in src["spv"]]) + "}"
			full_cpp += """
const KernelSourceImpl src__$KERNEL_NAME("$KERNEL_NAME", "$ENTRYPOINT", std::vector<uint32_t>($SPV));
kernel_ptr $KERNEL_NAME(const device_ptr& device) { LOAD_CACHED_KERNEL(device, src__$KERNEL_NAME) }
			""".replace("$KERNEL_NAME", src["name"]).replace("$ENTRYPOINT", src["entrypoint"]).replace("$SPV", spv_cpp_str)
			
			full_hpp += "\nkernel_ptr $KERNEL_NAME(const device_ptr& device);".replace("$KERNEL_NAME", src["name"])
		
		ns_end = """
} // $NAMESPACE
		""".replace("$NAMESPACE", namespace)
		
		full_cpp += ns_end
		full_hpp += ns_end
	t_ns_end = """
} // namespace kernels
} // namespace tart
	"""
	full_cpp += t_ns_end
	full_hpp += t_ns_end
	
	if persistent:
		raise NotImplementedError
	else:
		hpp_path = os.path.join(spv_directory, f"{source_file_name}.hpp")
		with open(hpp_path, "w") as f:
			f.write(full_hpp)
		with open(os.path.join(spv_directory, f"{source_file_name}.cpp"), "w") as f:
			f.write(full_cpp)
		assert os.path.exists(hpp_path)
	
	

if __name__ == "__main__":
	args = sys.argv[1:]
	kernel_source_filename = args[0]
	spv_directory = args[1]
	print(f"\n\nKERNEL_SOURCE_FILENAME: {kernel_source_filename}\n\n")
	print(f"\n\nSPV DIRECTORY: {spv_directory}\n\n")
	sources = {}
	spv_files = os.listdir(spv_directory)
	assert(len(spv_files) > 0)
	for kf in spv_files:
		if os.path.splitext(kf)[1] == ".spv":
			kp = os.path.join(spv_directory, kf)
			try:
				namespace, kernel_name, entry_point = os.path.splitext(kf)[0].split("____")
			except ValueError:
				raise ValueError(f"Failed to unpack {kf}")
			if not namespace in sources.keys():
				sources[namespace] = []
			sources[namespace].append({"name": kernel_name, "entrypoint": entry_point, "spv": read_spv(kp)})
	render_source_file(spv_directory, kernel_source_filename, sources)
