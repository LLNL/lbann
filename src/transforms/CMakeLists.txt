# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  normalize.cpp
  sample_normalize.cpp
  scale.cpp
  transform_pipeline.cpp
  )

add_subdirectory(vision)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
