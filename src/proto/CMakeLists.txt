# Create the LbannProto library
if (LBANN_HAS_PROTOBUF)

  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS lbann.proto)

  add_library(LbannProto ${PROTO_SRCS} ${PROTO_HDRS})
  target_link_libraries(LbannProto PUBLIC protobuf::libprotobuf)
  target_include_directories(LbannProto PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  # Install the library
  install(TARGETS LbannProto
    EXPORT LBANNTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

  # Install the newly built headers
  install(FILES ${PROTO_HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

endif (LBANN_HAS_PROTOBUF)

# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  init_image_data_readers.cpp
  proto_common.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
