set_full_path(_DIR_LBANN_CATCH2_TEST_FILES
  center_crop_test.cpp
  colorize_test.cpp
  grayscale_test.cpp
  horizontal_flip_test.cpp
  random_affine_test.cpp
  random_crop_test.cpp
  random_resized_crop_test.cpp
  random_resized_crop_with_fixed_aspect_ratio_test.cpp
  resize_test.cpp
  resized_center_crop_test.cpp
  to_lbann_layout_test.cpp
  transform_pipeline_test.cpp
  vertical_flip_test.cpp
  )

set(LBANN_CATCH2_TEST_FILES
  "${LBANN_CATCH2_TEST_FILES}" "${_DIR_LBANN_CATCH2_TEST_FILES}" PARENT_SCOPE)
