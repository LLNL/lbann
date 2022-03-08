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

# Defines the following variables:
#   - CNPY_FOUND
#   - CNPY_LIBRARIES
#   - CNPY_INCLUDE_DIRS
#
# Also creates an imported target CNPY

# Find the header
find_path(CNPY_INCLUDE_DIRS cnpy.h
  HINTS ${CNPY_DIR} $ENV{CNPY_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with CNPY header.")
find_path(CNPY_INCLUDE_DIRS cnpy.h)

# Find the library
find_library(CNPY_LIBRARY cnpy
  HINTS ${CNPY_DIR} $ENV{CNPY_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The CNPY library.")
find_library(CNPY_LIBRARY cnpy)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CNPY
  DEFAULT_MSG
  CNPY_LIBRARY CNPY_INCLUDE_DIRS)

# Setup the imported target
if (NOT TARGET CNPY::CNPY)
  add_library(CNPY::CNPY INTERFACE IMPORTED)
endif (NOT TARGET CNPY::CNPY)

# Set the include directories for the target
set_property(TARGET CNPY::CNPY APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CNPY_INCLUDE_DIRS})

# Set the link libraries for the target
set_property(TARGET CNPY::CNPY APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${CNPY_LIBRARY})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE CNPY_INCLUDE_DIRS)

# Set the libraries
set(CNPY_LIBRARIES CNPY::CNPY)
mark_as_advanced(FORCE CNPY_LIBRARY)
