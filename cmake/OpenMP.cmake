# Try finding OpenMP
find_package(OpenMP QUIET)

if(OPENMP_FOUND)

  # Status message
  message(STATUS "Found OpenMP")

  # Add C and C++ flags
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

  # LBANN has access to OpenMP
  set(LBANN_HAS_OPENMP TRUE)

endif()
