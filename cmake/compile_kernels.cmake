# ==================================================================================================
# Kernel compilation: uses tart-spirv-inliner to compile GLSL → SPIR-V and generate C++ headers.
# ==================================================================================================

# Path to the tart-spirv-inliner tool (built as part of Tart)
set(TART_SPIRV_INLINER ${CMAKE_BINARY_DIR}/tart/tools/tart-spirv-inliner/tart-spirv-inliner)

# Generate the kernel manifest (called once during configure)
function(tart_generate_kernel_manifest MANIFEST_PATH SOURCE_DIR SPV_DIR GLSLC_PATH)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

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

# Compile all kernels using tart-spirv-inliner (single process)
function(tart_compile_all_kernels MANIFEST_PATH)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # Create a custom command that generates headers via tart-spirv-inliner
    add_custom_command(
        OUTPUT "${MANIFEST_PATH}.compiled"
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/compile_all_kernels.py
                "${MANIFEST_PATH}" "${TART_SPIRV_INLINER}"
        COMMENT "Compiling BLAS kernels to SPIR-V and generating headers via tart-spirv-inliner..."
        DEPENDS tart-spirv-inliner
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        VERBATIM
    )
    add_custom_target(compile_all_kernels ALL
        DEPENDS "${MANIFEST_PATH}.compiled"
    )
    message(STATUS "Kernel compilation target registered (via tart-spirv-inliner)")
endfunction()

# Generate the kernel source .hpp/.cpp using tart-spirv-inliner
function(tart_create_kernel_source_file KERNEL_SOURCE_FILE_NAME SPV_DIR MANIFEST_PATH)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # Create placeholder files so target_sources works at configure time
    set(_HPP "${SPV_DIR}/${KERNEL_SOURCE_FILE_NAME}.hpp")
    set(_CPP "${SPV_DIR}/${KERNEL_SOURCE_FILE_NAME}.cpp")
    file(WRITE "${_HPP}" "// Placeholder - generated at build time")
    file(WRITE "${_CPP}" "// Placeholder - generated at build time")

    # Create the custom target that overwrites placeholders at build time
    add_custom_target("${KERNEL_SOURCE_FILE_NAME}_gen"
        COMMAND ${CMAKE_COMMAND} -E remove -f "${_HPP}" "${_CPP}"
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/compile_all_kernels.py
                "${MANIFEST_PATH}" "${TART_SPIRV_INLINER}"
        COMMENT "Generating kernel C++ sources via tart-spirv-inliner..."
        DEPENDS tart-spirv-inliner
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        VERBATIM
    )
    # The source generation depends on all kernels being compiled
    add_dependencies(${KERNEL_SOURCE_FILE_NAME}_gen compile_all_kernels)
endfunction()

macro(add_kernel_sources DEST_TARGET KERNEL_SOURCE_FILE_NAME SPV_DIR MANIFEST_PATH)
    tart_create_kernel_source_file(${KERNEL_SOURCE_FILE_NAME} "${SPV_DIR}" "${MANIFEST_PATH}")
    add_dependencies(${DEST_TARGET} ${KERNEL_SOURCE_FILE_NAME}_gen)
    target_sources(${DEST_TARGET} PRIVATE "${SPV_DIR}/${KERNEL_SOURCE_FILE_NAME}.cpp")
    target_include_directories(${DEST_TARGET} PUBLIC ${SPV_DIR})
endmacro()
