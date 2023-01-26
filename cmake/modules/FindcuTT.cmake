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
# This finds either the CUDA version (cuTT) or the HIP version (hipTT)
# of the cuTT library. The latter is just the hipified source of the
# former, so the file names (and APIs) are the same in both
# cases. Currently it's the user's responsibility to make sure that
# the one they find is the one suitable to their platform.
#
# Sets the following variables
#
#   cuTT_FOUND
#   cuTT_LIBRARIES
#
# Defines the following imported target:
#
#   cuTT::cuTT
#

find_path(cuTT_INCLUDE_PATH cutt.h
  DOC "The cuTT include directory.")

find_library(cuTT_LIBRARY cutt)

# Handle the find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  cuTT DEFAULT_MSG cuTT_LIBRARY cuTT_INCLUDE_PATH)

if (cuTT_FOUND)
  if (NOT TARGET cuTT::cuTT)
    add_library(cuTT::cuTT INTERFACE IMPORTED)
  endif ()
  target_link_libraries(cuTT::cuTT INTERFACE "${cuTT_LIBRARY}")
  target_include_directories(cuTT::cuTT INTERFACE "${cuTT_INCLUDE_PATH}")

  mark_as_advanced(cuTT_LIBRARY)
  mark_as_advanced(cuTT_INCLUDE_PATH)
endif ()

set(cuTT_LIBRARIES cuTT::cuTT)
