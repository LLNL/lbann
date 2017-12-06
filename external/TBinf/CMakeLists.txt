if (LBANN_HAS_PROTOBUF)

  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
    summary.proto
    event.proto
    )

  add_library(TBinf TBinf.cpp TBinf.hpp tbext.hpp ${PROTO_SRCS} ${PROTO_HDRS})
  target_link_libraries(TBinf PUBLIC protobuf::libprotobuf)
  target_include_directories(TBinf PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>)

  set(LBANN_HAS_TBINF TRUE)

endif (LBANN_HAS_PROTOBUF)
