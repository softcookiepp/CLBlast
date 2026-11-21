# ==================================================================================================
# Kernel compilation: generates a JSON manifest of all BLAS kernels, then compiles them in parallel.
# This replaces the old sequential execute_process loops that took ~13 minutes.
# ==================================================================================================

# Generate the kernel manifest (called once during configure)
function(tart_generate_kernel_manifest MANIFEST_PATH SOURCE_DIR SPV_DIR GLSLC_PATH)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    find_package(Vulkan COMPONENTS glslc REQUIRED)
    if(NOT GLSLC_PATH)
        set(GLSLC_PATH ${Vulkan_GLSLC_EXECUTABLE})
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/generate_all_kernels.py
                "${MANIFEST_PATH}" "${SOURCE_DIR}" "${SPV_DIR}" "${GLSLC_PATH}"
        RESULT_VARIABLE _gen_result
        OUTPUT_VARIABLE _gen_output
        ERROR_VARIABLE _gen_error
    )
    if(NOT _gen_result EQUAL 0)
        message(FATAL_ERROR "Failed to generate kernel manifest: ${_gen_error}")
    endif()
    message(STATUS "Kernel manifest generated: ${_gen_output}")
endfunction()

# Compile all kernels from the manifest (parallel, single Python process)
function(tart_compile_all_kernels MANIFEST_PATH)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # Create a custom command that compiles all kernels from the manifest
    add_custom_command(
        OUTPUT "${MANIFEST_PATH}.compiled"
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/compile_all_kernels.py "${MANIFEST_PATH}"
        COMMENT "Compiling BLAS kernels to SPIR-V (parallel)..."
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        VERBATIM
    )
    add_custom_target(compile_all_kernels ALL
        DEPENDS "${MANIFEST_PATH}.compiled"
    )
    message(STATUS "Parallel kernel compilation target registered")
endfunction()

# Generate the kernel source .hpp/.cpp from compiled SPIR-V files
function(tart_create_kernel_source_file KERNEL_SOURCE_FILE_NAME SPV_DIR)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # Create placeholder files so target_sources works at configure time
    set(_HPP "${SPV_DIR}/${KERNEL_SOURCE_FILE_NAME}.hpp")
    set(_CPP "${SPV_DIR}/${KERNEL_SOURCE_FILE_NAME}.cpp")
    file(WRITE "${_HPP}" "// Placeholder - generated at build time")
    file(WRITE "${_CPP}" "// Placeholder - generated at build time")

    # Create the custom target that overwrites placeholders at build time
    add_custom_target("${KERNEL_SOURCE_FILE_NAME}_gen"
        COMMAND ${CMAKE_COMMAND} -E remove -f "${_HPP}" "${_CPP}"
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/create_kernel_source_file.py
                "${KERNEL_SOURCE_FILE_NAME}" "${SPV_DIR}"
        COMMENT "Generating kernel C++ sources from SPIR-V..."
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        VERBATIM
    )
    # The source generation depends on all kernels being compiled
    add_dependencies(${KERNEL_SOURCE_FILE_NAME}_gen compile_all_kernels)
endfunction()

macro(add_kernel_sources DEST_TARGET KERNEL_SOURCE_FILE_NAME SPV_DIR)
    tart_create_kernel_source_file(${KERNEL_SOURCE_FILE_NAME} "${SPV_DIR}")
    add_dependencies(${DEST_TARGET} ${KERNEL_SOURCE_FILE_NAME}_gen)
    target_sources(${DEST_TARGET} PRIVATE "${SPV_DIR}/${KERNEL_SOURCE_FILE_NAME}.cpp")
    target_include_directories(${DEST_TARGET} PUBLIC ${SPV_DIR})
endmacro()
