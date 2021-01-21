################################################################################
## Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################

# Defines the following variables:
#   - FFTW_FOUND
#   - FFTW_VERSION
#   - FFTW_LIBRARIES
#
# Creates the following imported targets:
#   - FFTW::FFTW_FLOAT
#   - FFTW::FFTW_DOUBLE
#
# All targets are grouped under the imported target FFTW::FFTW.

# Try to find the float things
find_package(FFTW3f ${FFTW_FIND_VERSION} CONFIG QUIET
  HINTS ${FFTW_DIR} $ENV{FFTW_DIR}
  PATH_SUFFIXES lib64/cmake/fftw3f lib/cmake/fftw3f
  lib64/cmake/fftw3 lib/cmake/fftw3
  NO_DEFAULT_PATH)
find_package(FFTW3f ${FFTW_FIND_VERSION} CONFIG QUIET)

set(FFTW_FLOAT_FOUND)
set(FFTW_FLOAT_FOUND_BY_PKG_CONFIG)
set(FFTW_FLOAT_VERSION)
set(FFTW_FLOAT_IMPORTED_LIBRARY)
if (FFTW3f_FOUND)
  set(FFTW_FLOAT_FOUND ${FFTW3f_FOUND})
  set(FFTW_FLOAT_VERSION ${FFTW3f_VERSION})
  set(FFTW_FLOAT_IMPORTED_LIBRARY FFTW3::fftw3f)
else ()
  find_package(PkgConfig)
  if (PkgConfig_FOUND)
    pkg_check_modules(PC_FFTWF REQUIRED
      IMPORTED_TARGET GLOBAL
      fftw3f>=${FFTW_FIND_VERSION})
    if (NOT PC_FFTWF_FOUND)
      pkg_check_modules(PC_FFTWF REQUIRED
        IMPORTED_TARGET GLOBAL
        fftwf>=${FFTW_FIND_VERSION})
    endif ()
    if (PC_FFTWF_FOUND)
      set(FFTW_FLOAT_FOUND ${PC_FFTWF_FOUND})
      set(FFTW_FLOAT_FOUND_BY_PKG_CONFIG ${PC_FFTWF_FOUND})
      set(FFTW_FLOAT_VERSION ${PC_FFTWF_VERSION})
      set(FFTW_FLOAT_IMPORTED_LIBRARY PkgConfig::PC_FFTWF)
    endif ()
  endif (PkgConfig_FOUND)
endif ()

# Try to find the double things
if (NOT FFTW_FLOAT_FOUND_BY_PKG_CONFIG)
  find_package(FFTW3 ${FFTW_FIND_VERSION} CONFIG QUIET
    HINTS ${FFTW_DIR} $ENV{FFTW_DIR}
    PATH_SUFFIXES lib64/cmake/fftw3 lib/cmake/fftw3
    lib64/cmake/fftw3f lib/cmake/fftw3f
    NO_DEFAULT_PATH)
  find_package(FFTW3 ${FFTW_FIND_VERSION} CONFIG QUIET)
endif ()

set(FFTW_DOUBLE_FOUND)
set(FFTW_DOUBLE_VERSION)
set(FFTW_DOUBLE_IMPORTED_LIBRARY)
if (FFTW3_FOUND)
  set(FFTW_DOUBLE_FOUND ${FFTW3_FOUND})
  set(FFTW_DOUBLE_VERSION ${FFTW3_VERSION})
  set(FFTW_DOUBLE_IMPORTED_LIBRARY FFTW3::fftw3)
else ()
  find_package(PkgConfig)
  if (PkgConfig_FOUND)
    pkg_check_modules(PC_FFTW REQUIRED
      IMPORTED_TARGET GLOBAL
      fftw3>=${FFTW_FIND_VERSION})
    if (NOT PC_FFTW_FOUND)
      pkg_check_modules(PC_FFTW REQUIRED
        IMPORTED_TARGET GLOBAL
        fftw>=${FFTW_FIND_VERSION})
    endif ()
    if (PC_FFTW_FOUND)
      set(FFTW_DOUBLE_FOUND ${PC_FFTW_FOUND})
      set(FFTW_DOUBLE_VERSION ${PC_FFTW_VERSION})
      set(FFTW_DOUBLE_IMPORTED_LIBRARY PkgConfig::PC_FFTW)
    endif ()
  endif (PkgConfig_FOUND)
endif ()

set(FFTW_VERSION)
set(FOUND_OK_FFTW_LIB)

# Setup the float imported target
if (FFTW_FLOAT_FOUND)
  add_library(FFTW::FFTW_FLOAT INTERFACE IMPORTED)
  target_link_libraries(
    FFTW::FFTW_FLOAT INTERFACE ${FFTW_FLOAT_IMPORTED_LIBRARY})
  set(FOUND_OK_FFTW_LIB TRUE)
  set(FFTW_VERSION ${FFTW_FLOAT_VERSION})
endif (FFTW_FLOAT_FOUND)

# Setup the double imported target
if (FFTW_DOUBLE_FOUND)
  if (FFTW_FLOAT_FOUND)
    if (NOT FFTW_FLOAT_VERSION VERSION_EQUAL FFTW_DOUBLE_VERSION)
      message(WARNING
        "Found differently versioned FFTW libraries. FFTW(float) "
        "found with version ${FFTW_FLOAT_VERSION} and FFTW(double) "
        "found with version ${FFTW_DOUBLE_VERSION}.")
    endif ()
    if (FFTW_DOUBLE_VERSION VERSION_LESS FFTW_FLOAT_VERSION)
      set(FFTW_VERSION ${FFTW_DOUBLE_VERSION})
    endif ()
  else ()
    message(WARNING
      "FFTW(float) library not found. "
      "Only the double-precision library will be linked.")
    set(FFTW_VERSION ${FFTW_DOUBLE_VERSION})
  endif ()
  add_library(FFTW::FFTW_DOUBLE INTERFACE IMPORTED)
  target_link_libraries(
    FFTW::FFTW_DOUBLE INTERFACE ${FFTW_DOUBLE_IMPORTED_LIBRARY})
  set(FOUND_OK_FFTW_LIB TRUE)
endif (FFTW_DOUBLE_FOUND)

if (FOUND_OK_FFTW_LIB)
  add_library(FFTW::FFTW INTERFACE IMPORTED)
  target_link_libraries(FFTW::FFTW INTERFACE
    $<TARGET_NAME_IF_EXISTS:FFTW::FFTW_FLOAT>
    $<TARGET_NAME_IF_EXISTS:FFTW::FFTW_DOUBLE>)
endif (FOUND_OK_FFTW_LIB)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  FFTW DEFAULT_MSG FFTW_VERSION FOUND_OK_FFTW_LIB)
