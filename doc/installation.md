CLBlast-vk: Building and installing
================

This document describes how to compile, link, and install CLBlast-vk on various platforms. You can
either use a pre-built package or compile the library from source. For other information about
CLBlast-vk, see the [main README](../README.md).


Requirements
-------------

The pre-requisites for compilation of CLBlast-vk are kept as minimal as possible. A basic
compilation infrastructure is all you need:

* CMake version 3.10 or higher
* A C++17 compiler, for example:
  - GCC 8.0 or newer
  - Clang 8.0 or newer
  - AppleClang 11.0 or newer
  - MSVC (Visual Studio) 2019 or newer
* Vulkan SDK (for building Tart)
* Tart subproject (cloned via `git submodule`)


Linux / macOS compilation from source
-------------

Configuration can be done using CMake. On Linux and macOS systems with make, building is
straightforward. Here's an example of an out-of-source build using a command-line compiler and
make (starting from the root of the CLBlast-vk folder):

    git clone --recurse-submodules https://<repo>/CLBlast-vk.git
    cd CLBlast-vk
    mkdir build
    cd build
    cmake ..
    make
    sudo make install  # (optional)

A custom installation folder can be specified when calling CMake:

    cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/directory ..

Building a static version of the library instead of shared one (.dylib/.so) can be done by
disabling the `BUILD_SHARED_LIBS` option when calling CMake. For example:

    cmake -DBUILD_SHARED_LIBS=OFF ..


Windows compilation from source
-------------

When using Visual Studio 2019, the project-files can be generated as follows:

    mkdir build
    cd build
    cmake -G "Visual Studio 16 2019" -A x64 ..

To generate a static version of the library instead of a .dll, specify `-DBUILD_SHARED_LIBS=OFF`
when running cmake.


CMake options
-------------

The following CMake options are available:

* `BUILD_SHARED_LIBS` (default: ON) - Build a shared library (OFF for static)
* `SAMPLES` (default: OFF) - Build sample programs
* `TUNERS` (default: ON) - Build auto-tuning executables
* `CLIENTS` (default: OFF) - Build performance benchmark clients
* `TESTS` (default: OFF) - Build correctness tests
* `VULKAN_BDA` (default: OFF) - Enable Vulkan Buffer Device Address support
* `VERBOSE` (default: OFF) - Enable verbose diagnostic messages
* `TART_ENABLE_SHADER_COMPILERS` (default: ON) - Embed glslang compiler in Tart
* `TART_BUILD_STATIC` (default: OFF) - Build Tart as static library
