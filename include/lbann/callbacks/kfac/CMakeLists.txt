# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  kfac.hpp
  kfac_block.hpp
  kfac_block_bn.hpp
  kfac_block_fc_conv.hpp
  kfac_util.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
