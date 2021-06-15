# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  data_type_layer.cpp
  doer.cpp
  layer.cpp
  )

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/distconv_adapter.cpp")
  list(APPEND THIS_DIR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/data_type_distconv_adapter.cpp")
endif ()

# Add the subdirectories
add_subdirectory(activations)
add_subdirectory(cereal_registration)
add_subdirectory(image)
add_subdirectory(io)
add_subdirectory(learning)
add_subdirectory(loss)
add_subdirectory(math)
add_subdirectory(misc)
add_subdirectory(regularizers)
add_subdirectory(transform)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
