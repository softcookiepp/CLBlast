#!/usr/bin/env python3
"""
Generate a JSON manifest of all BLAS kernels to compile.
Called during CMake configure to produce a build-time manifest.

Usage:
    python generate_all_kernels.py <manifest_path> <source_dir> <spv_dir> <glslc_path>

Output:
    JSON manifest that can be consumed by compile_all_kernels.py
"""
import json
import os
import sys


def main():
    manifest_path = sys.argv[1]
    source_dir = sys.argv[2]
    spv_dir = sys.argv[3]
    glslc_path = sys.argv[4]

    binary_flag_options = [0, 1]
    vector_widths = [1, 2, 4, 8, 16]
    dtype_prefixes = ["F16", "F32", "F64", "C64", "C128"]

    kernels = []

    for dtype_prefix in dtype_prefixes:
        # LevelX kernels (im2col, col2im)
        for kernel in ["im2col_kernel_normal", "col2im_kernel_normal"]:
            kernels.append({
                "namespace": "blas",
                "name": f"{dtype_prefix}{kernel}",
                "entrypoint": "main",
                "language": "glsl",
                "source": os.path.join(source_dir, "kernels-vk/levelx", f"{kernel}.glsl"),
                "args": [f"-DDTYPE=DTYPE_{dtype_prefix}"],
            })

        # LevelX convgemm kernels (dual vector width)
        for kernel in ["convgemm_kernel", "convgemm_normal"]:
            for vwm in vector_widths:
                for vwn in vector_widths:
                    kernels.append({
                        "namespace": "blas",
                        "name": f"{dtype_prefix}{kernel}_vw{vwm}_vw{vwn}",
                        "entrypoint": "main",
                        "language": "glsl",
                        "source": os.path.join(source_dir, "kernels-vk/levelx", f"{kernel}.glsl"),
                        "args": [f"-DDTYPE=DTYPE_{dtype_prefix}", f"-DVWMD={vwm}", f"-DVWND={vwn}"],
                    })

        # transpose_fast (TRA_WPT)
        for vw in vector_widths:
            kernels.append({
                "namespace": "blas",
                "name": f"{dtype_prefix}transpose_fast_vw{vw}",
                "entrypoint": "main",
                "language": "glsl",
                "source": os.path.join(source_dir, "kernels-vk/level3/transpose_fast.glsl"),
                "args": [f"-DDTYPE=DTYPE_{dtype_prefix}", f"-DTRA_WPT={vw}"],
            })

        # Level3 no-arg kernels
        level3_noarg_no_x = [
            "convert_hermitian_upper", "convert_hermitian_lower",
            "convert_symmetric_upper", "convert_symmetric_lower",
            "convert_triangular_upper", "convert_triangular_lower",
            "level3_fill_matrix",
            "transpose_pad_matrix", "transpose_pad_matrix_batched", "transpose_pad_matrix_strided_batched",
            "transpose_matrix", "transpose_matrix_batched", "transpose_matrix_strided_batched",
            "copy_matrix_fast", "copy_pad_matrix", "copy_pad_matrix_batched", "copy_pad_matrix_strided_batched",
            "copy_matrix", "copy_matrix_batched", "copy_matrix_strided_batched",
            "invert_diagonal_block", "invert_triple_matmul", "invert_triple_matmul_part2",
        ]
        for kernel in level3_noarg_no_x:
            kernels.append({
                "namespace": "blas",
                "name": f"{dtype_prefix}{kernel}",
                "entrypoint": "main",
                "language": "glsl",
                "source": os.path.join(source_dir, "kernels-vk/level3", f"{kernel}.glsl"),
                "args": [f"-DDTYPE=DTYPE_{dtype_prefix}"],
            })

        # Level3 dual vector width kernels (xgemm_direct variants)
        for kernel in ["xgemm_direct", "xgemm_direct_batched", "xgemm_direct_strided_batched"]:
            for vwm in vector_widths:
                for vwn in vector_widths:
                    kernels.append({
                        "namespace": "blas",
                        "name": f"{dtype_prefix}{kernel}_vw{vwm}_vw{vwn}",
                        "entrypoint": "main",
                        "language": "glsl",
                        "source": os.path.join(source_dir, "kernels-vk/level3", f"{kernel}.glsl"),
                        "args": [f"-DDTYPE=DTYPE_{dtype_prefix}", f"-DVWM={vwm}", f"-DVWN={vwn}"],
                    })

        # Level3 subgroup dual vector width (xgemm variants)
        for kernel in ["xgemm", "xgemm_upper", "xgemm_lower", "xgemm_batched", "xgemm_strided_batched"]:
            for sub in binary_flag_options:
                kernels.append({
                    "namespace": "blas",
                    "name": f"{dtype_prefix}{kernel}_sub{sub}_vw1_vw1",
                    "entrypoint": "main",
                    "language": "glsl",
                    "source": os.path.join(source_dir, "kernels-vk/level3", f"{kernel}.glsl"),
                    "args": [f"-DDTYPE=DTYPE_{dtype_prefix}", "-DVWM=1", "-DVWN=1", f"-DSUBGROUP_OPERATIONS_SUPPORTED={sub}"],
                })

        # Vectorized Level1 kernels
        vw_kernels_level1 = [
            "xaxpy", "xaxpy_faster", "xaxpy_fastest",
            "xcopy", "xcopy_fast",
            "xscal", "xscal_fast",
            "xswap", "xswap_fast",
            "xhad", "xhad_faster", "xhad_fastest",
        ]
        for kernel in vw_kernels_level1:
            for vw in vector_widths:
                kernels.append({
                    "namespace": "blas",
                    "name": f"{dtype_prefix}{kernel}_vw{vw}",
                    "entrypoint": "main",
                    "language": "glsl",
                    "source": os.path.join(source_dir, "kernels-vk/level1", f"{kernel}.glsl"),
                    "args": [f"-DDTYPE=DTYPE_{dtype_prefix}", f"-DVW={vw}"],
                })

        # Level1 batched
        for kernel in ["xaxpy_batched"]:
            kernels.append({
                "namespace": "blas",
                "name": f"{dtype_prefix}{kernel}",
                "entrypoint": "main",
                "language": "glsl",
                "source": os.path.join(source_dir, "kernels-vk/level1", f"{kernel}.glsl"),
                "args": [f"-DDTYPE=DTYPE_{dtype_prefix}"],
            })

        # Level2 base kernels
        for kernel in ["xgemv", "xger", "xher", "xher2", "xtrsv_forward", "xtrsv_backward", "xtrsv_fill_vector"]:
            kernels.append({
                "namespace": "blas",
                "name": f"{dtype_prefix}{kernel}",
                "entrypoint": "main",
                "language": "glsl",
                "source": os.path.join(source_dir, "kernels-vk/level2", f"{kernel}.glsl"),
                "args": [f"-DDTYPE=DTYPE_{dtype_prefix}"],
            })

        # Level2 vectorized
        for kernel in ["xgemv_fast", "xgemv_fast_rot"]:
            for vw in vector_widths:
                kernels.append({
                    "namespace": "blas",
                    "name": f"{dtype_prefix}{kernel}_vw{vw}",
                    "entrypoint": "main",
                    "language": "glsl",
                    "source": os.path.join(source_dir, "kernels-vk/level2", f"{kernel}.glsl"),
                    "args": [f"-DDTYPE=DTYPE_{dtype_prefix}", f"-DVW={vw}"],
                })

        # Reduce kernels
        for kernel in [
            "xamax", "xamax_epilogue",
            "xasum", "xasum_epilogue",
            "xdot", "xdot_epilogue",
            "xnrm2", "xnrm2_epilogue",
        ]:
            kernels.append({
                "namespace": "blas",
                "name": f"{dtype_prefix}{kernel}",
                "entrypoint": "main",
                "language": "glsl",
                "source": os.path.join(source_dir, "kernels-vk/level1", f"{kernel}.glsl"),
                "args": [f"-DDTYPE=DTYPE_{dtype_prefix}"],
            })

    manifest = {
        "glslc": glslc_path,
        "output_dir": spv_dir,
        "kernels": kernels,
    }

    os.makedirs(os.path.dirname(manifest_path) if os.path.dirname(manifest_path) else ".", exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated manifest with {len(kernels)} kernels at {manifest_path}")


if __name__ == "__main__":
    main()
