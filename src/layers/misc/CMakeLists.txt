# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  covariance.cpp
  variance.cpp
  channelwise_mean.cpp
  argmax.cpp
  argmin.cpp
  one_hot.cpp
  )

if (LBANN_HAS_CUDA)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    covariance.cu
    variance.cu
    channelwise_mean.cu
    one_hot.cu
    )
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
