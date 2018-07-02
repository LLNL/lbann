# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  layer.cpp
  )

# Add the subdirectories
add_subdirectory(activations)
add_subdirectory(learning)
add_subdirectory(loss)
add_subdirectory(regularizers)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
