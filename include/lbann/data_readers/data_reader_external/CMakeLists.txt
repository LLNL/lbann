

# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  data_reader_external.hpp
  connection.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
