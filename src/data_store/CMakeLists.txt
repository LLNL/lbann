# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  generic_data_store.cpp
  data_store_csv.cpp
  data_store_image.cpp
  data_store_multi_images.cpp
  data_store_imagenet.cpp
  data_store_imagenet_patches.cpp
  data_store_merge_samples.cpp
  data_store_merge_features.cpp
  data_store_pilot2_molecular.cpp
  data_store_triplet.cpp
  jag_io.cpp
  jag_store.cpp
)

add_executable( jag_converter-bin jag_converter.cpp )
target_link_libraries(jag_converter-bin lbann )
set_target_properties(jag_converter-bin PROPERTIES OUTPUT_NAME jag_converter)


set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)

