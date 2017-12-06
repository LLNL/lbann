# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  cv_augmenter.cpp
  cv_colorizer.cpp
  cv_cropper.cpp
  cv_mean_extractor.cpp
  cv_normalizer.cpp
  cv_process.cpp
  cv_process_patches.cpp
  cv_utils.cpp
  data_reader.cpp
  data_reader_ascii.cpp
  data_reader_cifar10.cpp
  data_reader_csv.cpp
  data_reader_image.cpp
  data_reader_imagenet.cpp
  data_reader_imagenet_org.cpp
  data_reader_imagenet_patches.cpp
  data_reader_imagenet_single.cpp
  data_reader_merge_features.cpp
  data_reader_merge_samples.cpp
  data_reader_mnist.cpp
  data_reader_nci.cpp
  data_reader_numpy.cpp
  data_reader_pilot2_molecular.cpp
  data_reader_synthetic.cpp
  image_preprocessor.cpp
  image_utils.cpp
  )

# Add the subdirectories
add_subdirectory(patchworks)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
