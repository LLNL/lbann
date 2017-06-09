if(VTUNE_DIR)
  message(STATUS "Found VTune: ${VTUNE_DIR}")

  # Include directory.
  set(VTUNE_INCLUDE_DIRS ${VTUNE_DIR}/include)
  include_directories(${VTUNE_INCLUDE_DIRS})

  # Set VTune static library.
  set(VTUNE_STATIC_LIB ${VTUNE_DIR}/lib/libittnotify.a)

  # Set preprocessor flag.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLBANN_VTUNE")

  # LBANN has access to VTune.
  set(LBANN_HAS_VTUNE TRUE)

endif()