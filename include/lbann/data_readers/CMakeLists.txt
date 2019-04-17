# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  compound_data_reader.hpp
  cv_augmenter.hpp
  cv_colorizer.hpp
  cv_decolorizer.hpp
  cv_cropper.hpp
  cv_mean_extractor.hpp
  cv_normalizer.hpp
  cv_process.hpp
  cv_process_patches.hpp
  cv_transform.hpp
  cv_utils.hpp
  data_reader.hpp
  data_reader_ascii.hpp
  data_reader_cifar10.hpp
  data_reader_csv.hpp
  data_reader_image.hpp
  data_reader_imagenet.hpp
  data_reader_imagenet_patches.hpp
  data_reader_merge_features.hpp
  data_reader_merge_samples.hpp
  data_reader_mnist.hpp
  data_reader_moving_mnist.hpp
  data_reader_nci.hpp
  data_reader_numpy.hpp
  data_reader_numpy_npz.hpp
  data_reader_pilot2_molecular.hpp
  data_reader_python.hpp
  data_reader_synthetic.hpp
  image_preprocessor.hpp
  image_utils.hpp
  opencv.hpp
  opencv_extensions.hpp
  data_reader_multihead_siamese.hpp
  )

# Add the subdirectories
add_subdirectory(patchworks)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
