# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  buffered_data_coordinator.hpp
  data_coordinator.hpp
  data_coordinator_metadata.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
