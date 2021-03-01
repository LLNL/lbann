# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  argmax.cpp
  argmin.cpp
  channelwise_mean.cpp
  channelwise_softmax.cpp
  covariance.cpp
  dft_abs_builder.cpp
  dist_embedding.cpp
  mini_batch_index.cpp
  mini_batch_size.cpp
  one_hot.cpp
  uniform_hash.cpp
  variance.cpp
  )

if (LBANN_HAS_FFTW)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dft_abs.cpp")
endif ()

if (LBANN_HAS_GPU)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    covariance.cu
    dist_embedding.cu
    variance.cu
    channelwise_mean.cu
    channelwise_softmax.cu
    one_hot.cu
    uniform_hash.cu
    )
  if (LBANN_HAS_FFTW)
    list(APPEND THIS_DIR_CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dft_abs.cu")
  endif ()
endif ()

# Add the subdirectories
add_subdirectory(cereal_registration)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
