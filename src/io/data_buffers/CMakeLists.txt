# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  generic_io_buffer.cpp
  distributed_io_buffer.cpp
  partitioned_io_buffer.cpp
)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
