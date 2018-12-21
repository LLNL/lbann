# Create the LbannProto library
if (LBANN_HAS_PROTOBUF)

  # In its current state, LBANN does not make complicated use of
  # protobuf -- it's really just one file. This is the meat of the
  # implementation of "protobuf_generate_cpp" but it gives us a custom
  # command on which we can depend. Using this, when lbann.proto is
  # touched, CMake will rebuild the LbannProto library.
  set(PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/lbann.pb.cc")
  set(PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/lbann.pb.h")
  add_custom_command(
    COMMAND protobuf::protoc
    "--cpp_out=${DLL_EXPORT_DECL}${CMAKE_CURRENT_BINARY_DIR}"
    "-I" "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/lbann.proto"
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lbann.proto" protobuf::protoc
    COMMENT "Running protoc on lbann.proto"
    VERBATIM)

  add_custom_target(LbannProto_genSrc
    DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})

  add_library(LbannProto ${PROTO_SRCS} ${PROTO_HDRS})
  target_link_libraries(LbannProto PUBLIC protobuf::libprotobuf)
  target_include_directories(LbannProto SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
  target_include_directories(LbannProto PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  add_dependencies(LbannProto LbannProto_genSrc)

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

# Add the subdirectories
add_subdirectory(factories)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
