CLBlast: Building and installing
================

This document describes how to compile, link, and install CLBlast on various platforms. You can either use a pre-built package or compile the library from source. For other information about CLBlast, see the [main README](../README.md).


Requirements
-------------

The pre-requisites for compilation of CLBlast are kept as minimal as possible. A basic compilation infrastructure is all you need, no external dependencies are required. You'll need:

* CMake version 2.8.10 or higher
* A C++11 compiler, for example:
  - GCC 4.7.0 or newer
  - Clang 3.3 or newer
  - AppleClang 5.0 or newer
  - ICC 14.0 or newer
  - MSVC (Visual Studio) 2013 or newer
* Vulkan development headers or SDK
  - Follow the installation instructions [here](https://vulkan.lunarg.com/)


Using pre-built packages
-------------

Unlike the original CLBlast, you are out of luck here :c
You will have to build from source.

Cloning repo
-------------

    git clone https://github.com/softcookiepp/CLBlast.git
    cd CLBlast
    git submodule update --init --remote --recursive

Linux / macOS compilation from source
-------------

Before building, the submodule dependencies must be updated.
```
git submodule update --init --remote --recursive
```

Configuration can be done using CMake. On Linux and macOS systems with make, building is straightforward. Here's an example of an out-of-source build using a command-line compiler and make (starting from the root of the CLBlast folder):

    mkdir build
    cd build
    cmake ..
    make
    sudo make install  # (optional)

A custom installation folder can be specified when calling CMake:

    cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/directory ..

Building a static version of the library instead of shared one (.dylib/.so) can be done by disabling the `BUILD_SHARED_LIBS` option when calling CMake. For example:

    cmake -DBUILD_SHARED_LIBS=OFF ..

In case you run into segfaults with OpenCL programs (known to happen with the AMD APP), you can try the following (thanks to [kpot](https://github.com/CNugteren/CLBlast/issues/243#issuecomment-367277297)):

1. Use `-fPIC` or its analogue when compiling. In CMake you can do this by adding `set(CMAKE_POSITION_INDEPENDENT_CODE ON)` to the project config.

2. Forbid CMake to add RPATH entries to binaries. You can do this project-wise with `set(CMAKE_SKIP_BUILD_RPATH ON)` in CMake.


Windows compilation from source
-------------

~~When using Visual Studio 2015, the project-files can be generated as follows:~~

    ~~mkdir build~~
    ~~cd build~~
    ~~cmake -G "Visual Studio 14 Win64" ..~~

~~For another version, replace 14 with the appropriate version (12 for VS 2013, 15 for VS 2017). To generate a static version of the library instead of a .dll, specify `-DBUILD_SHARED_LIBS=OFF` when running cmake.~~
Building on Windows has not been tested at all. If somebody more familiar with the process would help, that would be very much appreciated!
