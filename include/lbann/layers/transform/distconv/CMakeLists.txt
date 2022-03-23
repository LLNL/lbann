set_full_path(THIS_DIR_HEADERS
  distconv_gather.hpp
  distconv_scatter.hpp
  distconv_nvshmem_vector_addressing.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
