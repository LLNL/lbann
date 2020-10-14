# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  argmax.hpp
  argmin.hpp
  channelwise_mean.hpp
  covariance.hpp
  dft_abs.hpp
  dft_abs_builder.hpp
  mini_batch_index.hpp
  mini_batch_size.hpp
  one_hot.hpp
  variance.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
