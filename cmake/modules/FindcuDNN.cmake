# Exports the following variables
#
#   CUDNN_FOUND
#   CUDNN_INCLUDE_PATH
#   CUDNN_LIBRARIES
#
# Exports the following IMPORTED target
#
#   cuda::cudnn
#

find_path(CUDNN_INCLUDE_PATH cudnn.h
  HINTS ${CUDNN_DIR} $ENV{CUDNN_DIR} ${cuDNN_DIR} $ENV{cuDNN_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Location of cudnn header."
  )
find_path(CUDNN_INCLUDE_PATH cudnn.h)

find_library(CUDNN_LIBRARY cudnn
  HINTS ${CUDNN_DIR} $ENV{CUDNN_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The cudnn library."
  )
find_library(CUDNN_LIBRARY cudnn)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuDNN
  DEFAULT_MSG CUDNN_LIBRARY CUDNN_INCLUDE_PATH)

if (NOT TARGET cuda::cudnn)

  add_library(cuda::cudnn INTERFACE IMPORTED)

  set_property(TARGET cuda::cudnn PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_PATH}")

  set_property(TARGET cuda::cudnn PROPERTY
    INTERFACE_LINK_LIBRARIES "${CUDNN_LIBRARY}")

endif (NOT TARGET cuda::cudnn)
