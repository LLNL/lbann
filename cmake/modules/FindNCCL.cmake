# Exports the following variables
#
#   NCCL_FOUND
#   NCCL_INCLUDE_PATH
#   NCCL_LIBRARY
#
# Exports the following IMPORTED targets:
#
#   cuda::nccl
#

find_path(NCCL_INCLUDE_PATH nccl.h
  HINTS ${NCCL_DIR} $ENV{NCCL_DIR} ${NCCL2_DIR} $ENV{NCCL2_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The location of NCCL headers."
  )
find_path(NCCL_INCLUDE_PATH nccl.h)

find_library(NCCL_LIBRARY nccl
  HINTS ${NCCL_DIR} $ENV{NCCL_DIR} ${NCCL2_DIR} $ENV{NCCL2_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The NCCL library."
  )
find_library(NCCL_LIBRARY nccl)

# Check the version. Note, this won't compile for NCCL1
set(_NCCL_VERSION_TEST_SRC "
#include <iostream>
#include <nccl.h>

int main()
{
    std::cout << NCCL_MAJOR << \".\" << NCCL_MINOR << \".\" << NCCL_PATCH << std::endl;
    return EXIT_SUCCESS;
}
")
file(WRITE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx"
  "${_NCCL_VERSION_TEST_SRC}\n")

try_run(_NCCL_RUN_RESULT _NCCL_COMPILE_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx
  CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${CUDA_INCLUDE_DIRS};${NCCL_INCLUDE_PATH}"
  RUN_OUTPUT_VARIABLE _NCCL_VERSION_STRING
  COMPILE_OUTPUT_VARIABLE _NCCL_COMPILE_OUTPUT
  )

# Assume that if it didn't compile, we have NCCL1
if (NOT _NCCL_COMPILE_RESULT)
  message(${_NCCL_COMPILE_OUTPUT})
  set(_NCCL_VERSION_STRING 1.0.0)
endif ()

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_PATH
  VERSION_VAR _NCCL_VERSION_STRING)

# Setup the imported target
if (NOT TARGET cuda::nccl)

  add_library(cuda::nccl INTERFACE IMPORTED)

  set_property(TARGET cuda::nccl PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${NCCL_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})

  set_property(TARGET cuda::nccl PROPERTY
    INTERFACE_LINK_LIBRARIES ${NCCL_LIBRARY})

endif (NOT TARGET cuda::nccl)
