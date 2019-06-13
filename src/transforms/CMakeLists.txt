# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  normalize.cpp
  repack_HWC_to_CHW_layout.cpp
  sample_normalize.cpp
  scale.cpp
  scale_and_translate.cpp
  transform_pipeline.cpp
  )

add_subdirectory(vision)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
