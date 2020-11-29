# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  channelwise_scale_bias.cpp
  embedding.cpp
  entrywise_scale_bias.cpp
  gru.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
