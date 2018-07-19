if (LBANN_HAS_PROTOBUF)

  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
    summary.proto
    event.proto
    )

  add_library(TBinf TBinf.cpp TBinf.hpp tbext.hpp ${PROTO_SRCS} ${PROTO_HDRS})
  target_link_libraries(TBinf PUBLIC protobuf::libprotobuf)
  target_include_directories(TBinf SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
  target_include_directories(TBinf PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  # Install the library
  install(TARGETS TBinf
    EXPORT LBANNTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

  # Install the newly created headers
  install(FILES ${PROTO_HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  set(LBANN_HAS_TBINF TRUE PARENT_SCOPE)

endif (LBANN_HAS_PROTOBUF)
