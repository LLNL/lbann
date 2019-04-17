# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  generic_data_store.hpp
  data_store_conduit.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
