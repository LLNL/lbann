# Try finding OpenMP
find_package(OpenMP QUIET)

# Here is a hack to make it work on OS X with the current Homebrew installation
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(NOT OPENMP_FOUND)
     set(OpenMP_C_FLAGS "-fopenmp -I/usr/local/include/libiomp/ -L/usr/local/lib/")
     set(OpenMP_CXX_FLAGS "-fopenmp -I/usr/local/include/libiomp/ -L/usr/local/lib/")
  endif()
endif()

# Status message
message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")

# Add C and C++ flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
