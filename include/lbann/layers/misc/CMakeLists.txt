# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  covariance.hpp
  variance.hpp
  channelwise_mean.hpp
  mini_batch_index.hpp
  mini_batch_size.hpp
  argmax.hpp
  argmin.hpp
  one_hot.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
