# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  base_convolution.cpp
  channelwise_fully_connected.cpp
  channelwise_scale_bias.cpp
  channelwise_scale_bias_builder.cpp
  convolution.cpp
  deconvolution.cpp
  entrywise_scale_bias.cpp
  embedding.cpp
  embedding_builder.cpp
  fully_connected.cpp
  gru.cpp
  )

if (LBANN_HAS_GPU)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    channelwise_scale_bias.cu
    embedding.cu
    entrywise_scale_bias.cu
    )
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
