# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  categorical_accuracy.cpp
  cross_entropy.cpp
  entrywise.cpp
  l1_norm.cpp
  l2_norm2.cpp
  mean_absolute_error.cpp
  mean_squared_error.cpp
  top_k_categorical_accuracy.cpp
  )

if (LBANN_HAS_GPU)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    categorical_accuracy.cu
    cross_entropy.cu
    entrywise.cu
    l1_norm.cu
    l2_norm2.cu
    mean_absolute_error.cu
    mean_squared_error.cu
    top_k_categorical_accuracy.cu
    )
endif ()

# Add the subdirectories
add_subdirectory(cereal_registration)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
