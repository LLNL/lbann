# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  cv_augmenter.cpp
  cv_colorizer.cpp
  cv_cropper.cpp
  cv_decolorizer.cpp
  cv_mean_extractor.cpp
  cv_normalizer.cpp
  cv_process.cpp
  cv_process_patches.cpp
  cv_resizer.cpp
  cv_subtractor.cpp
  cv_transform.cpp
  cv_utils.cpp
  data_reader.cpp
  data_reader_ascii.cpp
  data_reader_cifar10.cpp
  data_reader_csv.cpp
  data_reader_image.cpp
  data_reader_imagenet.cpp
  data_reader_imagenet_patches.cpp
  data_reader_jag.cpp
  data_reader_jag_conduit.cpp
  data_reader_merge_features.cpp
  data_reader_merge_samples.cpp
  data_reader_mesh.cpp
  data_reader_mnist.cpp
  data_reader_moving_mnist.cpp
  data_reader_nci.cpp
  data_reader_numpy.cpp
  data_reader_numpy_npz.cpp
  data_reader_pilot2_molecular.cpp
  data_reader_synthetic.cpp
  data_reader_multi_images.cpp
  data_reader_mnist_siamese.cpp
  data_reader_multihead_siamese.cpp
  data_reader_python.cpp
  offline_patches_npz.cpp
  image_preprocessor.cpp
  image_utils.cpp
  numpy_conduit_converter.cpp 
  data_reader_numpy_npz_conduit.cpp
  )

# Add the subdirectories
add_subdirectory(patchworks)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
