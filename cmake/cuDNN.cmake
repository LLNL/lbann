if(DEFINED CMAKE_CUDNN_DIR)

  # Status message
  message(STATUS "Found cuDNN: ${CMAKE_CUDNN_DIR}")

  # cuDNN header files and libraries
  set(CUDNN_INCLUDE_DIRS "${CMAKE_CUDNN_DIR}/include")
  include_directories(${CUDNN_INCLUDE_DIRS})
  set(CUDNN_LIBRARIES "${CMAKE_CUDNN_DIR}/lib64/libcudnn.so")

  # Add preprocessor flag for cuDNN  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_CUDNN")

  # LBANN has access to cuDNN
  set(LBANN_HAS_CUDNN TRUE)

endif()