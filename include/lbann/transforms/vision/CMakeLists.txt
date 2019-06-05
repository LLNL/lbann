# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  adjust_brightness.hpp
  adjust_contrast.hpp
  adjust_saturation.hpp
  center_crop.hpp
  colorize.hpp
  color_jitter.hpp
  cutout.hpp
  grayscale.hpp
  horizontal_flip.hpp
  normalize_to_lbann_layout.hpp
  random_affine.hpp
  random_crop.hpp
  random_resized_crop.hpp
  random_resized_crop_with_fixed_aspect_ratio.hpp
  resize.hpp
  resized_center_crop.hpp
  to_lbann_layout.hpp
  vertical_flip.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
