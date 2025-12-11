import pytart
import os
import numpy as np

inst = pytart.Instance()
dev = inst.create_device(0)

with open("test64.comp", "r") as f:
	src = f.read()

module = dev.compile_glsl(src)
pipeline = dev.create_pipeline(module, "main")

a = np.random.randn(256).astype(np.float64)
a_buf = dev.allocate_buffer(a.nbytes)
a_buf.copy_in(a)

push = np.zeros(2).astype(np.int32)
push[0] = 256
push.view(np.float32)[1] = 2.0

sequence = dev.create_sequence()
sequence.record_pipeline(pipeline, [1, 1, 1], [a_buf], push)
dev.submit_sequence(sequence)
dev.sync()

print(a[0:16])
a_buf.copy_out(a)
print(a[0:16])
