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
#   Clara_FOUND
#   Clara_LIBRARIES
#   Clara_INCLUDE_PATH
#
# Also creates an imported target clara::clara

# Find the header
find_path(CLARA_INCLUDE_PATH clara.hpp
  HINTS ${CLARA_DIR} $ENV{CLARA_DIR} ${Clara_DIR} $ENV{Clara_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_path(CLARA_INCLUDE_PATH clara.hpp)

# Handle the find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Clara DEFAULT_MSG CLARA_INCLUDE_PATH)

# Build the imported target
if (NOT TARGET clara::clara)
  add_library(clara::clara INTERFACE IMPORTED)
endif()

set_property(TARGET clara::clara
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${CLARA_INCLUDE_PATH})

# Set the last of the output variables
set(CLARA_LIBRARIES clara::clara)

# Cleanup
mark_as_advanced(FORCE CLARA_INCLUDE_PATH)
