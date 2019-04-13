# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  data_store_conduit.cpp
)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
