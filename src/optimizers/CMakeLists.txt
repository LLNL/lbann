# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  adagrad.cpp
  adam.cpp
  hypergradient_adam.cpp
  optimizer.cpp
  rmsprop.cpp
  sgd.cpp
  )

if (LBANN_HAS_CUDA)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    adam.cu
    sgd.cu
    )
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
