# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  layer.hpp
  data_type_layer.hpp
  )

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/distconv_adapter.hpp")
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/data_type_distconv_adapter.hpp")
endif ()

# Add the subdirectories
add_subdirectory(activations)
add_subdirectory(image)
add_subdirectory(io)
add_subdirectory(learning)
add_subdirectory(loss)
add_subdirectory(math)
add_subdirectory(misc)
add_subdirectory(regularizers)
add_subdirectory(transform)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
