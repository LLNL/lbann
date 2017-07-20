if(cuDNN_DIR)

  # Status message
  message(STATUS "Found cuDNN: ${cuDNN_DIR}")

  # cuDNN header files and libraries
  set(cuDNN_INCLUDE_DIRS ${cuDNN_DIR}/include)
  include_directories(${cuDNN_INCLUDE_DIRS})
  set(cuDNN_LIBRARIES ${cuDNN_DIR}/lib64/${CMAKE_SHARED_LIBRARY_PREFIX}cudnn${CMAKE_SHARED_LIBRARY_SUFFIX})

  # Add preprocessor flag for cuDNN  
  add_definitions(-D__LIB_CUDNN)

  # LBANN has access to cuDNN
  set(LBANN_HAS_CUDNN TRUE)

endif()