# Try finding CUDA
find_package(CUDA QUIET)

if(CUDA_FOUND)

  # Status message
  message(STATUS "Found CUDA (version ${CUDA_VERSION_STRING}): ${CUDA_TOOLKIT_ROOT_DIR}")

  # Include CUDA header files
  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(/usr/workspace/wsb/brain/nccl2/include/)

  set(cuBLAS_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/${CMAKE_SHARED_LIBRARY_PREFIX}cublas${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(NVTX_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/${CMAKE_SHARED_LIBRARY_PREFIX}nvToolsExt${CMAKE_SHARED_LIBRARY_SUFFIX})  
  set(nccl2_LIBRARIES /usr/workspace/wsb/brain/nccl2/lib/${CMAKE_SHARED_LIBRARY_PREFIX}nccl${CMAKE_SHARED_LIBRARY_SUFFIX})

  # Add preprocessor flag for CUDA
  add_definitions(-D__LIB_CUDA)

  # set the minimum architecture
  set(CUDA_NVCC_FLAGS -arch;sm_30;-std=c++11)

  # LBANN has access to CUDA
  set(LBANN_HAS_CUDA TRUE)

endif()
