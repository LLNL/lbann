################################################################################
## Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

# Exports the following variables
#
#   LBANN_HAS_VTUNE
#   VTUNE_INCLUDE_DIR
#   VTUNE_STATIC_LIB
#

find_path(VTUNE_INCLUDE_PATH libittnotify.h
  HINTS ${VTUNE_DIR} $ENV{VTUNE_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The location of VTune headers.")
find_path(VTUNE_INCLUDE_PATH libittnotify.h)

find_library(VTUNE_STATIC_LIB libittnotify.a
  HINTS ${VTUNE_DIR} $ENV{VTUNE_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The location of VTune Static library.")
find_library(VTUNE_STATIC_LIB libittnotify.a)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VTune
  REQUIRED_VARS VTUNE_INCLUDE_PATH VTUNE_STATIC_LIB)

if (VTUNE_INCLUDE_PATH AND VTUNE_STATIC_LIB)
  if (NOT TARGET vtune::vtune)
    add_library(vtune::vtune STATIC IMPORTED)
  endif ()

  set_target_properties(vtune::vtune PROPERTIES
    IMPORTED_LOCATION "${VTUNE_STATIC_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${VTUNE_INCLUDE_PATH}")

  set(VTUNE_LIBRARIES vtune::vtune)
endif ()
