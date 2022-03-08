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

# Output variables
#
#   HWLOC_FOUND
#   HWLOC_LIBRARIES
#   HWLOC_INCLUDE_PATH
#
# Also creates an imported target HWLOC::hwloc

if (MPI_FOUND)
  list(APPEND _TMP_MPI_LIBS "${MPI_C_LIBRARIES}" "${MPI_CXX_LIBRARIES}")
  foreach (lib IN LISTS _TMP_MPI_LIBS)
    get_filename_component(_TMP_MPI_LIB_DIR "${lib}" DIRECTORY)
    list(APPEND _TMP_MPI_LIBRARY_DIRS ${_TMP_MPI_LIB_DIR})
  endforeach ()

  if (_TMP_MPI_LIBRARY_DIRS)
    list(REMOVE_DUPLICATES _TMP_MPI_LIBRARY_DIRS)
  endif ()
endif (MPI_FOUND)

# Find the library
find_library(HWLOC_LIBRARY hwloc
  HINTS ${HWLOC_DIR} $ENV{HWLOC_DIR} ${_TMP_MPI_LIBRARY_DIRS}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH)
find_library(HWLOC_LIBRARY hwloc)

# Find the header
find_path(HWLOC_INCLUDE_PATH hwloc.h
  HINTS ${HWLOC_DIR} $ENV{HWLOC_DIR}
  ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_path(HWLOC_INCLUDE_PATH hwloc.h)

# Handle the find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  HWLOC DEFAULT_MSG HWLOC_LIBRARY HWLOC_INCLUDE_PATH)

# Build the imported target
if (NOT TARGET HWLOC::hwloc)
  add_library(HWLOC::hwloc INTERFACE IMPORTED)
endif()

set_property(TARGET HWLOC::hwloc
  PROPERTY INTERFACE_LINK_LIBRARIES ${HWLOC_LIBRARY})

if (NOT "/usr/include" STREQUAL "${HWLOC_INCLUDE_PATH}")
  set_property(TARGET HWLOC::hwloc
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${HWLOC_INCLUDE_PATH})
endif ()

# Set the last of the output variables
set(HWLOC_LIBRARIES HWLOC::hwloc)

# Cleanup
mark_as_advanced(FORCE HWLOC_INCLUDE_PATH)
mark_as_advanced(FORCE HWLOC_LIBRARY)
