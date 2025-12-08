import os

if os.path.exists("../src"):
	os.chdir("..")
assert os.path.exists("src/kernels-vk")
os.chdir("src/kernels-vk")

def compile_glsl():
	raise NotImplementedError
	
def create_inline(path):
	out_path = path + ".inl"
	with open(path, "r") as f:
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
