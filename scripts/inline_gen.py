import os

if os.path.exists("../src"):
	os.chdir("..")
assert os.path.exists("src/kernels-vk")
print("found kernel path")
os.chdir("src/kernels-vk")

def compile_glsl():
	raise NotImplementedError
	
def create_inline(path):
	raise NotImplementedError
	out_path = path + ".inl"
	with open(path, "r") as f:
		body = f.read()
	
	
def generate_inlines(d):
	for fn in os.listdir(d):
		if "glsl" in fn:
			if os.path.splitext(fn)[1] == ".inl": continue
			path = os.path.join(d, fn)
			create_inline(path)

for d in os.listdir():
	if not os.path.isdir(d): continue
	generate_inlines(d)
generate_inlines(".")
