#message(FATAL_ERROR "JPEG-turbo not yet implemented.")
set(JPEG-TURBO_URL https://github.com/libjpeg-turbo/libjpeg-turbo
  CACHE STRING "The URL from which to clone LIBJPEG-TURBO")

set(JPEG-TURBO_TAG "1.5.2" 
  CACHE STRING "The git tag to checkout for LIBJPEG-TURBO")

# Where to install LIBJPEG-TURBO
set (JPEG-TURBO_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of LIBJPEG-TURBO.")

set(JPEG-TURBO_MAKE_JOBS "8"
  CACHE INTERNAL "The number of make jobs to run. Max by default.")


find_program(GNU_MAKE_PROGRAM make)
#if (GNU_MAKE_PROGRAM)
#  mark_as_advanced(GNU_MAKE_PROGRAM)
#else ()
#  message(FATAL_ERROR "GNU-compatible \'make\' program not found."
#    "LibJpeg-Turbo requires \'make\'.")
#endif ()

configure_file(jpeg-turbo_configure.sh.in
  ${CMAKE_CURRENT_BINARY_DIR}/tmp/jpeg-turbo_configure.sh @ONLY)

# Handle the install of LIBJPEG-TURBO
include(ExternalProject)
ExternalProject_Add(JPEG-TURBO
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${JPEG-TURBO_URL}
  GIT_TAG ${JPEG-TURBO_TAG}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${JPEG-TURBO_CMAKE_INSTALL_PREFIX}
  CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tmp/jpeg-turbo_configure.sh
  BUILD_COMMAND  ${GNU_MAKE_PROGRAM} -j${JPEG-TURBO_MAKE_JOBS} 
  
  )

  set(JPEG-TURBO_DIR ${JPEG-TURBO_CMAKE_INSTALL_PREFIX}
    CACHE INTERNAL "The install prefix of JPEG-TURBO.")
