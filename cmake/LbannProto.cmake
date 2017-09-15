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
  # Don't duplicate custom targets.
  if (NOT TARGET protobuf_built)
    if(LBANN_BUILT_PROTOBUF)
      add_custom_target(protobuf_built DEPENDS project_protobuf)
    else()
      add_custom_target(protobuf_built)
    endif()
  endif()

  add_custom_command(
    OUTPUT ${LBANN_PROTO_DIR}/lbann.pb.cc
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=${PROJECT_SOURCE_DIR}/src/proto --cpp_out=${LBANN_PROTO_DIR}/ ${PROJECT_SOURCE_DIR}/src/proto/lbann.proto
    DEPENDS protobuf_built ${PROJECT_SOURCE_DIR}/src/proto/lbann.proto
    WORKING_DIRECTORY ${LBANN_PROTO_DIR}
  )

  # Install header file
  install(FILES ${LBANN_PROTO_DIR}/lbann.pb.h DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})

  # Build and install library 
  add_library(LbannProto SHARED ${LBANN_PROTO_DIR}/lbann.pb.cc)
  install(
    TARGETS LbannProto
    EXPORT LbannProto
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  set(LBANN_BUILT_LBANN_PROTO TRUE)

endif()

set(LBANN_HAS_LBANN_PROTO TRUE)
include_directories(SYSTEM ${LBANN_PROTO_DIR})

