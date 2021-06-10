# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  base_convolution.hpp
  channelwise_scale_bias.hpp
  channelwise_fully_connected.hpp
  convolution.hpp
  deconvolution.hpp
  embedding.hpp
  entrywise_scale_bias.hpp
  fully_connected.hpp
  fully_connected_cuda.hpp
  gru.hpp
  )

if (LBANN_HAS_DISTCONV)
  add_subdirectory(activations)
endif()

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
