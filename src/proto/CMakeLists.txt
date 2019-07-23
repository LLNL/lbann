# Create the LbannProto library
if (LBANN_HAS_PROTOBUF)

  # In its current state, LBANN does not make complicated use of
  # protobuf -- it's really just one file. This is the meat of the
  # implementation of "protobuf_generate_cpp" but it gives us a custom
  # command on which we can depend. Using this, when lbann.proto is
  # touched, CMake will rebuild the LbannProto library.
  set_full_path(PROTO_INPUTS lbann.proto callbacks.proto)

  foreach (proto IN LISTS PROTO_INPUTS)
    get_filename_component(name "${proto}" NAME_WE)
    list(APPEND PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${name}.pb.cc")
    list(APPEND PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/${name}.pb.h")
    list(APPEND PROTO_PY "${CMAKE_CURRENT_BINARY_DIR}/${name}_pb2.py")
  endforeach ()
  add_custom_command(
    COMMAND protobuf::protoc
    "--cpp_out=${CMAKE_CURRENT_BINARY_DIR}"
    "--python_out=${CMAKE_CURRENT_BINARY_DIR}"
    "-I" "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROTO_INPUTS}"
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${PROTO_PY}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lbann.proto" protobuf::protoc
    COMMENT "Running protoc on lbann.proto"
    COMMAND_EXPAND_LISTS
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

  # Install the Python module into the site-packages directory
  install(FILES ${PROTO_PY} DESTINATION
                ${PYTHON_INSTALL_PREFIX}/${CMAKE_INSTALL_PYTHONDIR})

endif (LBANN_HAS_PROTOBUF)

# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  init_image_data_readers.cpp
  proto_common.cpp
  helpers.cpp
  )

# Add the subdirectories
add_subdirectory(factories)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
