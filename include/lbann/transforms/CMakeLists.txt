# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  normalize.hpp
  sample_normalize.hpp
  scale.hpp
  transform.hpp
  transform_pipeline.hpp
  )

add_subdirectory(vision)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
