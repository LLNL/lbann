include(ExternalProject)

# Include Protocol Buffers dependency
if(NOT LBANN_HAS_PROTOBUF)
  include(protobuf)
endif()

if(LBANN_HAS_PROTOBUF)

  # TBinf source and header files
  set(TBINF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/TBinf)

  # Build location
  set(TBINF_PROTO_DIR ${PROJECT_BINARY_DIR}/external/tbinf)
  file(MAKE_DIRECTORY ${TBINF_PROTO_DIR})

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
    OUTPUT ${TBINF_PROTO_DIR}/summary.pb.cc ${TBINF_PROTO_DIR}/event.pb.cc
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=./ --cpp_out=${TBINF_PROTO_DIR}/ summary.proto event.proto
    DEPENDS protobuf_built
    WORKING_DIRECTORY ${TBINF_SOURCE_DIR}
  )

  # Include header files
  set(TBinf_INCLUDE_DIRS ${TBINF_PROTO_DIR} ${TBINF_SOURCE_DIR})
  include_directories(SYSTEM ${TBinf_INCLUDE_DIRS})

  # Build library
  add_library(TBinf SHARED ${TBINF_SOURCE_DIR}/TBinf.cpp ${TBINF_PROTO_DIR}/summary.pb.cc ${TBINF_PROTO_DIR}/event.pb.cc)
  target_link_libraries(TBinf ${PROTOBUF_LIBRARIES})

  # Add preprocessor flag for Tensorboard interface
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HAVE_TBINF")

  # LBANN has access to Tensorboard interface
  set(LBANN_HAS_TBINF TRUE)

endif()
