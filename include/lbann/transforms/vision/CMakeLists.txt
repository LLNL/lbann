# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  center_crop.hpp
  colorize.hpp
  grayscale.hpp
  horizontal_flip.hpp
  normalize_to_lbann_layout.hpp
  random_affine.hpp
  random_crop.hpp
  random_resized_aspect_ratio_crop.hpp
  random_resized_crop.hpp
  resize.hpp
  resized_center_crop.hpp
  to_lbann_layout.hpp
  vertical_flip.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
