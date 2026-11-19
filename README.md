
CLBlast-vk: CLBlast fork with Vulkan backend
================

This is a fork of [CLBlast](https://github.com/CNugteren/CLBlast) that replaces its OpenCL backend
with a custom Vulkan compute engine called **Tart**. CLBlast is a tuned BLAS library; this fork
produces a Vulkan-based BLAS library (`libclblast`) that targets Vulkan compute shaders written in
GLSL.

This project is **not** intended to be merged back into upstream CLBlast. It is a standalone fork
that will eventually be renamed or factored into another library.


Getting started
-------------

CLBlast-vk can be compiled with minimal dependencies (apart from Vulkan and Tart) in the usual
CMake-way, e.g.:

    mkdir build && cd build
    cmake ..
    make

Detailed instructions for various platforms can be found [here](doc/installation.md).

CLBlast-vk requires Tart, a Vulkan-based GPGPU framework, as a git submodule. Clone with:

    git clone --recurse-submodules https://<repo>/CLBlast-vk.git

After building, any of CLBlast's BLAS routines can be called directly. Use the Vulkan header:

    #include <clblast_vk.h>

The API is kept as close as possible to the Netlib BLAS, cuBLAS, and clBLAS APIs. The available
routines and the required arguments are described in `clblast_vk.h` and the included
[API documentation](doc/api.md). For an overview of the supported routines, see
[here](doc/routines.md).

To get started quickly, a couple of stand-alone example programs are included in the `samples`
subfolder. They can optionally be compiled using the CMake infrastructure by providing the
`-DSAMPLES=ON` flag, for example as follows:

    cmake -DSAMPLES=ON ..

Afterwards, you can optionally read more about running proper [benchmarks](doc/benchmarking.md)
and [tuning the library](doc/tuning.md).


Full documentation
-------------

More detailed documentation is available in separate files:

* [Building and installing](doc/installation.md)
* [Supported routines overview](doc/routines.md)
* [Performance measuring and benchmarking](doc/benchmarking.md)
* [Tuning for better performance](doc/tuning.md)
* [Testing the library for correctness](doc/testing.md)
* [More details on the GEMM kernel](doc/details_gemm.md)
* [More details on the convolution implementation](doc/details_conv.md)
* [Glossary with some terms explained](doc/glossary.md)
* [Frequently asked questions (FAQ) and their answers](doc/faq.md)


Known issues
-------------

Known issues:

* Correctness issues on Intel Arc A770 and several other devices with version 1.6.2 or lower
  (depends on the device). Upgrade to version 1.6.3 or newer.

* Routines returning an integer are currently not properly tested for half-precision FP16:
  IHAMAX/IHAMIN/IHMAX/IHMIN

* Half-precision FP16 tests might sometimes fail based on order of multiplication,
  i.e. (a * b) * c != (c * b) * a


Contributing
-------------

Contributions are welcome in the form of tuning results for Vulkan devices or pull requests.
See [the contributing guidelines](CONTRIBUTING.md) for more details.


Support us
-------------

This project started in March 2015 as an evenings and weekends free-time project next to a
full-time job for Cedric Nugteren. You can find contact information on the [website of the main
author](http://cnugteren.github.io).
