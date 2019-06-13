# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  center_crop.cpp
  colorize.cpp
  grayscale.cpp
  horizontal_flip.cpp
  normalize_to_lbann_layout.cpp
  random_affine.cpp
  random_crop.cpp
  random_resized_crop.cpp
  random_resized_crop_with_fixed_aspect_ratio.cpp
  resize.cpp
  resized_center_crop.cpp
  to_lbann_layout.cpp
  vertical_flip.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
