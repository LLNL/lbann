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
find_package(FFTW3f ${PACKAGE_FIND_VERSION} CONFIG QUIET
  HINTS ${FFTW_DIR} $ENV{FFTW_DIR}
  PATH_SUFFIXES lib64/cmake/fftw3f lib/cmake/fftw3f
  lib64/cmake/fftw3 lib/cmake/fftw3
  NO_DEFAULT_PATH)
find_package(FFTW3f ${PACKAGE_FIND_VERSION} CONFIG QUIET)

# Try to find the double things
find_package(FFTW3 ${PACKAGE_FIND_VERSION} CONFIG QUIET
  HINTS ${FFTW_DIR} $ENV{FFTW_DIR}
  PATH_SUFFIXES lib64/cmake/fftw3 lib/cmake/fftw3
  lib64/cmake/fftw3f lib/cmake/fftw3f
  NO_DEFAULT_PATH)
find_package(FFTW3 ${PACKAGE_FIND_VERSION} CONFIG QUIET)

set(FOUND_OK_FFTW_LIB)
set(FFTW_VERSION)

# Setup the float imported target
if (FFTW3f_FOUND)
  add_library(FFTW::FFTW_FLOAT INTERFACE IMPORTED)
  target_link_libraries(FFTW::FFTW_FLOAT INTERFACE FFTW3::fftw3f)
  set(FOUND_OK_FFTW_LIB TRUE)
  set(FFTW_VERSION "${FFTW3f_VERSION}")
endif (FFTW3f_FOUND)

# Setup the double imported target
if (FFTW3_FOUND)
  # Check for version consistency.
  set(_SKIP_FFTW_DOUBLE)
  if (FFTW_VERSION)
    if (NOT FFTW3_VERSION VERSION_EQUAL FFTW_VERSION)
      message(WARNING
        "FFTW double-precision library found with different "
        "version than single precision. Found ${FFTW3_VERSION}; "
        "expected ${FFTW_VERSION}. Only single-precision library "
        "will be accepted.")
      set(_SKIP_FFTW_DOUBLE TRUE)
    endif ()
  else ()
    message(WARNING
      "Single-precision FFTW library not found; only "
      "double-precision library will be used.")
    set(FFTW_VERSION "${FFTW3f_VERSION}")
  endif (FFTW_VERSION)

  if (NOT _SKIP_FFTW_DOUBLE)
    add_library(FFTW::FFTW_DOUBLE INTERFACE IMPORTED)
    target_link_libraries(FFTW::FFTW_DOUBLE INTERFACE FFTW3::fftw3)
    set(FOUND_OK_FFTW_LIB TRUE)
  endif (NOT _SKIP_FFTW_DOUBLE)
endif (FFTW3_FOUND)

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
