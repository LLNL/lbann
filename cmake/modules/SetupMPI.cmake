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

# This module configures MPI and ensures the library is setup properly

find_package(MPI REQUIRED COMPONENTS C CXX)

if (NOT TARGET MPI::MPI_CXX)
  add_library(MPI::MPI_CXX INTERFACE IMPORTED)
  if (MPI_CXX_COMPILE_FLAGS)
    separate_arguments(_MPI_CXX_COMPILE_OPTIONS UNIX_COMMAND
      "${MPI_CXX_COMPILE_FLAGS}")
    set_property(TARGET MPI::MPI_CXX PROPERTY
      INTERFACE_COMPILE_OPTIONS "${_MPI_CXX_COMPILE_OPTIONS}")
  endif()

  if (MPI_CXX_LINK_FLAGS)
    separate_arguments(_MPI_CXX_LINK_LINE UNIX_COMMAND
      "${MPI_CXX_LINK_FLAGS}")
  endif()

  set_property(TARGET MPI::MPI_CXX PROPERTY
    INTERFACE_LINK_LIBRARIES "${_MPI_CXX_LINK_LINE}")

  set_property(TARGET MPI::MPI_CXX APPEND PROPERTY
    LINK_FLAGS "${_MPI_CXX_LINK_LINE}")

  set_property(TARGET MPI::MPI_CXX PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")

endif (NOT TARGET MPI::MPI_CXX)

# Patch around pthread on Lassen
get_property(_TMP_MPI_CXX_COMPILE_FLAGS TARGET MPI::MPI_CXX
  PROPERTY INTERFACE_COMPILE_OPTIONS)
set_property(TARGET MPI::MPI_CXX PROPERTY
  INTERFACE_COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:CXX>:${_TMP_MPI_CXX_COMPILE_FLAGS}>)

get_property(_TMP_MPI_LINK_LIBRARIES TARGET MPI::MPI_CXX
  PROPERTY INTERFACE_LINK_LIBRARIES)
foreach(lib IN LISTS _TMP_MPI_LINK_LIBRARIES)
  if ("${lib}" MATCHES "-Wl*")
    list(APPEND _MPI_LINK_FLAGS "${lib}")
  else()
    list(APPEND _MPI_LINK_LIBRARIES "${lib}")
  endif ()
endforeach()

#set_property(TARGET MPI::MPI_CXX PROPERTY LINK_FLAGS ${_MPI_LINK_FLAGS})
set_property(TARGET MPI::MPI_CXX
  PROPERTY INTERFACE_LINK_LIBRARIES ${_MPI_LINK_LIBRARIES})
