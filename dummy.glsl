#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, std430) buffer a_buf { vec2 a[]; };
layout(binding = 0, std430) buffer b_buf { vec2 b[]; };

void main()
{
	uint gid = gl_WorkGroupID.x;
	for (uint i = 0; i < 2; i += 1)
	{
		a[gid][i] = 10.0;
		b[gid][i] = 10.0;
	}
}
