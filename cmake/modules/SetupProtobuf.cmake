set(PROTOBUF_MIN_VERSION "3.0.0")
if (PROTOBUF_DIR)
  list(APPEND CMAKE_LIBRARY_PATH ${PROTOBUF_DIR}/lib)
  list(APPEND CMAKE_INCLUDE_PATH ${PROTOBUF_DIR}/include)
  list(APPEND CMAKE_PREFIX_PATH ${PROTOBUF_DIR})
endif ()
find_package(Protobuf ${Protobuf_MIN_VERSION} REQUIRED)
if (PROTOBUF_DIR)
  list(REMOVE_ITEM CMAKE_LIBRARY_PATH ${PROTOBUF_DIR}/lib)
  list(REMOVE_ITEM CMAKE_INCLUDE_PATH ${PROTOBUF_DIR}/include)
  list(REMOVE_ITEM CMAKE_PREFIX_PATH ${PROTOBUF_DIR})
endif ()

# Setup the imported target for old versions of CMake
if (NOT TARGET protobuf::libprotobuf)
  add_library(protobuf::libprotobuf INTERFACE IMPORTED)
  set_property(TARGET protobuf::libprotobuf PROPERTY
    INTERFACE_LINK_LIBRARIES "${PROTOBUF_LIBRARIES}")

  set_property(TARGET protobuf::libprotobuf PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${PROTOBUF_INCLUDE_DIRS}")
endif ()

set(LBANN_HAS_PROTOBUF ${Protobuf_FOUND})
