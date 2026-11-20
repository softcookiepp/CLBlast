
# ==================================================================================================
# Kernel source collection for CLBlast-vk AOT kernel cache
# ==================================================================================================
# After compile_kernels.cmake runs (via tart_add_kernel_set), generated .hpp/.cpp bundles
# exist at ${TART_KERNEL_DIRECTORY}/<name>.hpp and <name>.cpp.
# Collect all generated .cpp files and add them to the clblast library.
# ==================================================================================================

# Ensure TART_KERNEL_DIRECTORY is set
if(NOT DEFINED TART_KERNEL_DIRECTORY)
    set(TART_KERNEL_DIRECTORY ${PROJECT_BINARY_DIR}/kernels)
endif()

# Collect all generated kernel source files from all kernel sets
set(_KERN_SOURCES)
foreach(_KERNEL_SET ${TART_AOT_KERNEL_SETS})
    list(APPEND _KERN_SOURCES
        ${TART_KERNEL_DIRECTORY}/${_KERNEL_SET}.hpp
        ${TART_KERNEL_DIRECTORY}/${_KERNEL_SET}.cpp
    )
endforeach()

# Generate a wrapper header that includes all kernel declarations
set(_WRAPPER_HEADER "${CMAKE_BINARY_DIR}/all-kernels-includes.hpp")
file(WRITE "${_WRAPPER_HEADER}" "#pragma once\n")
foreach(_KERNEL_SET ${TART_AOT_KERNEL_SETS})
    file(APPEND "${_WRAPPER_HEADER}" "#include <${TART_KERNEL_DIRECTORY}/${_KERNEL_SET}.hpp>\n")
endforeach()

# Copy shared GLSL include files to build directory so relative includes resolve
file(GLOB GLSL_INCLUDES
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/common.glsl"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/dtype-enum.glsl"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level1/*.glsl"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level2/*.glsl"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/level3/*.glsl"
)
foreach(_GLSL_FILE ${GLSL_INCLUDES})
    file(COPY "${_GLSL_FILE}" DESTINATION "${CMAKE_BINARY_DIR}/kernels/")
endforeach()

# Create a custom target for kernel sources (for IDE visibility)
add_custom_target("all-kernel-sources" SOURCES ${_KERN_SOURCES})

# Wire generated kernel .cpp files into clblast library
target_sources(clblast PRIVATE ${_KERN_SOURCES})

# Ensure tart-spirv-inliner is built before clblast
add_dependencies(clblast tart-spirv-inliner)
