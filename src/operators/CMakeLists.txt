# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  data_type_operator.cpp
  elementwise_operator.cpp
  operator.cpp
)

# Add the subdirectories
add_subdirectory(math)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
