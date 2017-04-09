#include(ExternalProject)

# Include Protocol Buffers dependency
if(NOT LBANN_HAS_PROTOBUF)
  include(protobuf)
endif()

if(NOT LBANN_PROTO_DIR OR FORCE_LBANN_PROTO_BUILD)

  # Build location
  set(LBANN_PROTO_DIR ${PROJECT_BINARY_DIR}/lbann_proto)
  file(MAKE_DIRECTORY ${LBANN_PROTO_DIR})

  # Generate source and header files with protocol buffer compiler
  if(LBANN_BUILT_PROTOBUF)
    add_custom_target(protobuf_built DEPENDS project_protobuf)
  else()
    add_custom_target(protobuf_built)
  endif()

  add_custom_command(
    OUTPUT ${LBANN_PROTO_DIR}/lbann.pb.cc
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=${PROJECT_SOURCE_DIR}/src/proto --cpp_out=${LBANN_PROTO_DIR}/ ${PROJECT_SOURCE_DIR}/src/proto/lbann.proto
    COMMAND ${CMAKE_COMMAND} -E copy
      ${LBANN_PROTO_DIR}/lbann.pb.h 
      ${CMAKE_INSTALL_PREFIX}/include
    DEPENDS protobuf_built
    WORKING_DIRECTORY ${LBANN_PROTO_DIR}
  )

  # install header file
  install(FILES lbann.pb.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

  # Build library 
  add_library(LbannProto SHARED ${LBANN_PROTO_DIR}/lbann.pb.cc)
  install(FILES ${CMAKE_INSTALL_PREFIX}/libLbannProto.so DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

  set(LBANN_BUILT_LBANN_PROTO TRUE)

endif()


set(LbannProto_LIBRARIES ${CMAKE_INSTALL_PREFIX}/${CMAKE_SHARED_LIBRARY_PREFIX}LbannProto${CMAKE_SHARED_LIBRARY_SUFFIX})

set(LBANN_HAS_LBANN_PROTO TRUE)
