import os

if os.path.exists("../src"):
	os.chdir("..")
assert os.path.exists("src/kernels-vk")
KERNEL_DIR = "src/kernels-vk"
INL_DIR = "src/kernels-vk-inline"

def compile_glsl():
	raise NotImplementedError
	
def create_inline(in_path, out_path):
	with open(in_path, "r") as f:
		body = f.read()
	
	# uncomment the string beginning
	body = body.replace('//R"(\n', 'R"(\n')
	# then uncomment the string ending
	body = body.replace('//)"\n', ')"\n')
	# finally, get rid of any include/version statements
	new_lines = []
	for line in body.split("\n"):
		if not ("#version" in line or "#include" in line):
			new_lines.append(line)
	
	new_body = "\n".join(new_lines)
	if os.path.exists(out_path):
		# check to see if contents are identical.
		# if they are, leave it alone so the file isn't flagged as changed
		with open(out_path, "r") as f:
			if f.read() == new_body:
				return
		os.remove(out_path)
	with open(out_path, "w") as f:
		f.write(new_body)
	
def generate_inlines(in_d, out_d):
	if not os.path.exists(out_d):
		os.mkdir(out_d)
	for fn in os.listdir(in_d):
		if "glsl" in fn:
			#if os.path.splitext(fn)[1] == ".inl": continue
			in_path = os.path.join(in_d, fn)
			out_path = os.path.join(out_d, fn + ".inl")
			create_inline(in_path, out_path)

for d in os.listdir(KERNEL_DIR):
	in_d = os.path.join(KERNEL_DIR, d)
	out_d = os.path.join(INL_DIR, d)
	if not os.path.isdir(in_d): continue
	generate_inlines(in_d, out_d)
generate_inlines(KERNEL_DIR, INL_DIR)
