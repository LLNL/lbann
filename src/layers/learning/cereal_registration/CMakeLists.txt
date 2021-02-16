# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  channelwise_fully_connected.cpp
  channelwise_scale_bias.cpp
  convolution.cpp
  deconvolution.cpp
  embedding.cpp
  entrywise_scale_bias.cpp
  fully_connected.cpp
  gru.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
