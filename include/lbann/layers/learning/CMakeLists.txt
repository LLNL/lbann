# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  base_convolution.hpp
  convolution.hpp
  deconvolution.hpp
  fully_connected.hpp
  fully_connected_cuda.hpp
  learning.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
