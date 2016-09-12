# Try finding OpenMP
find_package(OpenMP QUIET)

# Status message
message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")

# Add C and C++ flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
