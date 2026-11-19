

set(TART_KERNEL_DIRECTORY ${PROJECT_BINARY_DIR}/kernels CACHE STRING "this again")
set(TART_SCRIPTS_DIRECTORY "${PROJECT_SOURCE_DIR}/scripts" CACHE STRING "and this again too")

# Collect all generated kernel source files for the given namespaces.
# The compile_kernels.cmake script generates one kernel set per namespace.
macro(tart_add_kernel_sources TARGET NAME NAMESPACES)
    set(_KERN_SOURCES)
    foreach(_NS ${NAMESPACES})
        set(_KERN_SOURCES ${_KERN_SOURCES}
            ${TART_KERNEL_DIRECTORY}/${NAME}_${_NS}.hpp
            ${TART_KERNEL_DIRECTORY}/${NAME}_${_NS}.cpp
        )
    endforeach()
    add_custom_target("${NAME}" SOURCES ${_KERN_SOURCES})
    target_sources(${TARGET} PRIVATE ${_KERN_SOURCES})
endmacro()
