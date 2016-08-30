if(DEFINED CMAKE_CUDNN_DIR)

  # Status message
  message(STATUS "Found cuDNN: ${CMAKE_CUDNN_DIR}")

  # cuDNN header files and libraries
  include_directories("${CMAKE_CUDNN_DIR}/include")
  link_directories("${CMAKE_CUDNN_DIR}/lib64")
  set(CUDNN_LIBRARIES "-lcudnn")

  # Add preprocessor flag for cuDNN  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_CUDNN")

  # LBANN has access to cuDNN
  set(LBANN_HAS_CUDNN TRUE)

endif()