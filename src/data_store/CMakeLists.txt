# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  generic_data_store.cpp
  data_store_csv.cpp
  data_store_image.cpp
  data_store_multi_images.cpp
  data_store_imagenet.cpp
  data_store_merge_samples.cpp
  data_store_merge_features.cpp
  data_store_pilot2_molecular.cpp
  data_store_triplet.cpp
)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)

