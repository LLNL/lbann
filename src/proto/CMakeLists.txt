# Create the LbannProto library
if (LBANN_HAS_PROTOBUF)

  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS lbann.proto)

  add_library(LbannProto ${PROTO_SRCS} ${PROTO_HDRS})
  target_link_libraries(LbannProto PUBLIC protobuf::libprotobuf)
  target_include_directories(LbannProto PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>)
  set(LBANN_HAS_TBINF TRUE)

endif (LBANN_HAS_PROTOBUF)

# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  init_image_data_readers.cpp
  proto_common.cpp
  )
#list(APPEND THIS_DIR_SOURCES ${PROTO_SRCS})


# Propagate the files up the tree
#set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
