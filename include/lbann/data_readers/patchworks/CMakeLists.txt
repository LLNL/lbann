# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  patchworks.hpp
  patchworks_ROI.hpp
  patchworks_common.hpp
  patchworks_patch_descriptor.hpp
  patchworks_stats.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
