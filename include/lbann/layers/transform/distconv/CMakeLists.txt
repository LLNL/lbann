set_full_path(THIS_DIR_HEADERS
  distconv_gather.hpp
  distconv_scatter.hpp
  scatter_nvshmem.hpp
  gather_nvshmem.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)