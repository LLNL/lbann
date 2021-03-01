# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  argmax.cpp
  argmin.cpp
  channelwise_mean.cpp
  channelwise_softmax.cpp
  covariance.cpp
  dist_embedding.cpp
  mini_batch_index.cpp
  mini_batch_size.cpp
  one_hot.cpp
  variance.cpp
  )

if (LBANN_HAS_GPU)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/uniform_hash.cpp")
endif ()

if (LBANN_HAS_FFTW)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dft_abs.cpp")
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
