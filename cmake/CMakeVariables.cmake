# ==================================================================================================
# Kernel source collection for CLBlast-vk AOT kernel cache
# ==================================================================================================
# Collects generated .hpp/.cpp bundles and wires them into the clblast library.
# Also copies shared GLSL include files to build directory for shader compilation.
#
# Usage in CMakeLists.txt:
#   include(${PROJECT_SOURCE_DIR}/cmake/CMakeVariables.cmake)
#   set(TART_AOT_KERNEL_SETS "blas-kernels")
#   tart_collect_kernel_sources(clblast)
# ==================================================================================================

# Ensure TART_KERNEL_DIRECTORY is set
if(NOT DEFINED TART_KERNEL_DIRECTORY)
    set(TART_KERNEL_DIRECTORY ${PROJECT_BINARY_DIR}/kernels)
endif()

# Ensure TART_AOT_KERNEL_SETS is set (empty means no kernel sets to process)
if(NOT DEFINED TART_AOT_KERNEL_SETS)
    set(TART_AOT_KERNEL_SETS "")
endif()

# Collect all generated kernel source files from all kernel sets
set(_KERN_SOURCES)
foreach(_KERNEL_SET ${TART_AOT_KERNEL_SETS})
    list(APPEND _KERN_SOURCES
        ${TART_KERNEL_DIRECTORY}/${_KERNEL_SET}.hpp
        ${TART_KERNEL_DIRECTORY}/${_KERNEL_SET}.cpp
    )
endforeach()

# Copy shared GLSL include files to build directory so relative includes resolve
file(GLOB GLSL_INCLUDES
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/common.glsl"
    "${PROJECT_SOURCE_DIR}/src/kernels-vk/dtype-enum.glsl"
)
foreach(_GLSL_FILE ${GLSL_INCLUDES})
    file(COPY "${_GLSL_FILE}" DESTINATION "${PROJECT_BINARY_DIR}/spv/")
endforeach()
