# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  init_image_data_readers.hpp
  proto_common.hpp
  helpers.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
