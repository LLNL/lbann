include(ExternalProject)

# Options
option(FORCE_OPENCV_BUILD "OpenCV: force build" OFF)

# Try finding OpenCV
if(NOT FORCE_OPENCV_BUILD)
  find_package(OpenCV QUIET HINTS ${OpenCV_DIR})
endif()

# Check if OpenCV has been found
if(OpenCV_FOUND AND NOT FORCE_OPENCV_BUILD)

  # Status message
  message(STATUS "Found OpenCV (version ${OpenCV_VERSION}): ${OpenCV_DIR}")

else()

  # Git repository URL and tag
  if(NOT OPENCV_URL)
    set(OPENCV_URL https://github.com/opencv/opencv.git)
  endif()
  if(NOT OPENCV_TAG)
    set(OPENCV_TAG "2.4.13")
  endif()
  message(STATUS "Will pull OpenCV (tag ${OPENCV_TAG}) from ${OPENCV_URL}")

  # OpenCV build options
  if(NOT OPENCV_BUILD_TYPE)
    set(OPENCV_BUILD_TYPE ${CMAKE_BUILD_TYPE})
  endif()
  option(OPENCV_BUILD_DOCS "OpenCV: Create build rules for OpenCV Documentation" OFF)
  option(OPENCV_BUILD_EXAMPLES "OpenCV: Build all examples" OFF)
  option(OPENCV_BUILD_PERF_TESTS "OpenCV: Build performance tests" OFF)
  option(OPENCV_BUILD_TESTS "OpenCV: Build accuracy & regression tests" OFF)
  option(OPENCV_WITH_CUDA "OpenCV: Include NVidia Cuda Runtime support" OFF)
  option(OPENCV_WITH_IPP "OpenCV: Include Intel IPP support" OFF) # Causes a hash mismatch error when downloading
  option(OPENCV_WITH_GPHOTO "OpenCV: Include gPhoto2 library support" OFF) # Causes a compilation error

  # Download and build location
  set(OPENCV_SOURCE_DIR ${PROJECT_BINARY_DIR}/download/opencv/source)
  set(OPENCV_BINARY_DIR ${PROJECT_BINARY_DIR}/download/opencv/build)

  # Get OpenCV from Git repository and build
  ExternalProject_Add(project_OpenCV
    PREFIX          ${CMAKE_INSTALL_PREFIX}
    TMP_DIR         ${OPENCV_BINARY_DIR}/tmp
    STAMP_DIR       ${OPENCV_BINARY_DIR}/stamp
    GIT_REPOSITORY  ${OPENCV_URL}
    GIT_TAG         ${OPENCV_TAG}
    SOURCE_DIR      ${OPENCV_SOURCE_DIR}
    BINARY_DIR      ${OPENCV_BINARY_DIR}
    BUILD_COMMAND   ${CMAKE_MAKE_PROGRAM} -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}
    INSTALL_DIR     ${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}
    CMAKE_ARGS
      -D CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -D CMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE}
      -D CMAKE_BUILD_TYPE=${OPENCV_BUILD_TYPE}
      -D WITH_CUDA=${OPENCV_WITH_CUDA}
      -D WITH_IPP=${OPENCV_WITH_IPP}
      -D WITH_GPHOTO2=${OPENCV_WITH_GPHOTO2}
      -D BUILD_DOCS=${OPENCV_BUILD_DOCS}
      -D BUILD_EXAMPLES=${OPENCV_BUILD_EXAMPLES}
      -D BUILD_PERF_TESTS=${OPENCV_BUILD_PERF_TESTS}
      -D BUILD_TESTS=${OPENCV_BUILD_TESTS}
  )

  # Get install directory
  set(OpenCV_DIR ${CMAKE_INSTALL_PREFIX})

  # Get header files
  set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}/include)

  # Get libraries
  set(OpenCV_LIBRARIES ${OpenCV_DIR}/lib/libopencv_highgui.so ${OpenCV_DIR}/lib/libopencv_core.so)

  # LBANN has built OpenCV
  set(LBANN_BUILT_OPENCV TRUE)

endif()

# Include header files
include_directories(${OpenCV_INCLUDE_DIRS})

# Add preprocessor flag for OpenCV
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_OPENCV")

# LBANN has access to OpenCV
set(LBANN_HAS_OPENCV TRUE)
