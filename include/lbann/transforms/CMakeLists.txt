# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  normalize.hpp
  repack_HWC_to_CHW_layout.hpp
  sample_normalize.hpp
  scale.hpp
  scale_and_translate.hpp
  transform.hpp
  transform_pipeline.hpp
  )

if (LBANN_HAS_OPENCV)
  add_subdirectory(vision)
endif ()

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
