if(cuDNN_DIR)

  # Status message
  message(STATUS "Found cuDNN: ${cuDNN_DIR}")

  # cuDNN header files and libraries
  set(cuDNN_INCLUDE_DIRS "${cuDNN_DIR}/include")
  include_directories(${cuDNN_INCLUDE_DIRS})
  set(cuDNN_LIBRARIES "${cuDNN_DIR}/lib64/libcudnn.so")

  # Add preprocessor flag for cuDNN  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_CUDNN")

  # LBANN has access to cuDNN
  set(LBANN_HAS_CUDNN TRUE)

endif()