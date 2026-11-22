#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tart::tart" for configuration ""
set_property(TARGET tart::tart APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(tart::tart PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtart.so.0.1.0"
  IMPORTED_SONAME_NOCONFIG "libtart.so.0.1.0"
  )

list(APPEND _cmake_import_check_targets tart::tart )
list(APPEND _cmake_import_check_files_for_tart::tart "${_IMPORT_PREFIX}/lib/libtart.so.0.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
