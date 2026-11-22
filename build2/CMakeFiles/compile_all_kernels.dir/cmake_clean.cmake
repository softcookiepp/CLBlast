file(REMOVE_RECURSE
  "CMakeFiles/compile_all_kernels"
  "spv/kernels-manifest.json.compiled"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/compile_all_kernels.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
