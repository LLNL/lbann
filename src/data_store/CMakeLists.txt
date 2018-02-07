# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  generic_data_store.cpp
  data_store_image.cpp
  data_store_imagenet.cpp
)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
