set(JPEG-TURBO_URL https://github.com/ndryden/allreduces
  CACHE STRING "The URL from which to clone aluminum")

set(ALUMINUM_TAG "master" 
  CACHE STRING "The git tag to checkout for aluminum")

# Where to install LIBJPEG-TURBO
set (ALUMINUM_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of aluminum.")

set(ALUMINUM_MAKE_JOBS "8"
  CACHE INTERNAL "The number of make jobs to run. Max by default.")


find_program(GNU_MAKE_PROGRAM make)
#if (GNU_MAKE_PROGRAM)
#  mark_as_advanced(GNU_MAKE_PROGRAM)
#else ()
#  message(FATAL_ERROR "GNU-compatible \'make\' program not found."
#    "LibJpeg-Turbo requires \'make\'.")
#endif ()


# Handle the install of LIBJPEG-TURBO
include(ExternalProject)
ExternalProject_Add(JPEG-TURBO
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${ALUMINUM_URL}
  GIT_TAG ${ALUMINUM_TAG}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${JPEG-TURBO_CMAKE_INSTALL_PREFIX}
  BUILD_COMMAND  ${GNU_MAKE_PROGRAM} -j${JPEG-TURBO_MAKE_JOBS} 
  
  )

  set(ALUMINUM_DIR ${ALUMINUM_CMAKE_INSTALL_PREFIX}
    CACHE INTERNAL "The install prefix of aluminum.")
