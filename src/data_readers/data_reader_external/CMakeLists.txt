# Create the LbannCommunicationProto library
if (LBANN_HAS_PROTOBUF)

  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS data_reader_communication.proto)

  add_library(LbannCommunicationProto ${PROTO_SRCS} ${PROTO_HDRS})
  target_link_libraries(LbannCommunicationProto PUBLIC protobuf::libprotobuf)
  target_include_directories(LbannCommunicationProto SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
  target_include_directories(LbannCommunicationProto PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  # Install the library
  install(TARGETS LbannCommunicationProto
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
  data_reader_external.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
