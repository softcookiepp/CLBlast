import os
import sys

def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)

if __name__ == "__main__":
	print(sys.argv)
	args = sys.argv[1:]
	compiler = args[0]
	namespace = args[1]
	kernel_name = args[2]
	entry_point = args[3]
	source_language = args[4]
	source_path = args[5]
	assert(os.path.exists(source_path))
	
	kernel_directory = args[6]
	mkdir(kernel_directory)
	compiler_args = []
	if len(args) > 6:
		compiler_args = args[7:]
	
	spv_path = os.path.join(kernel_directory, f"{namespace}____{kernel_name}____{entry_point}.spv")
	if not os.path.exists(spv_path):
		print(compiler_args)
		if source_language == "glsl":
			cmd = f"{compiler} --target-spv=spv1.5 -fshader-stage=compute {source_path} {' '.join(compiler_args)} -o {spv_path}"
			print(cmd)
			assert os.system(cmd) == 0
		else:
			raise NotImplementedError
	
