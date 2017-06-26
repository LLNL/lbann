# Try finding CUDA
find_package(CUDA QUIET)

if(CUDA_FOUND)

  # Status message
  message(STATUS "Found CUDA (version ${CUDA_VERSION_STRING}): ${CUDA_TOOLKIT_ROOT_DIR}")

  # Include CUDA header files
  include_directories(${CUDA_INCLUDE_DIRS})

  set(cuBLAS_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/${CMAKE_SHARED_LIBRARY_PREFIX}cublas${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(NVTX_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/${CMAKE_SHARED_LIBRARY_PREFIX}nvToolsExt${CMAKE_SHARED_LIBRARY_SUFFIX})  

  # Add preprocessor flag for CUDA
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_CUDA")

  # LBANN has access to CUDA
  set(LBANN_HAS_CUDA TRUE)

endif()
