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
#   - ZSTR_FOUND
#   - ZSTR_INCLUDE_DIRS
#
# Also creates an imported target ZSTR::ZSTR
#
# I can't find any interesting version information in the headers. So
# good luck with that.

# Find the header
find_path(ZSTR_INCLUDE_DIR zstr.hpp
  HINTS ${ZSTR_DIR} $ENV{ZSTR_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with ZSTR header.")
find_path(ZSTR_INCLUDE_DIR zstr.hpp)

set(ZSTR_INCLUDE_DIRS "${ZSTR_INCLUDE_DIR}"
  CACHE STRING "The list of paths required for ZSTR usage" FORCE)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZSTR
  DEFAULT_MSG
  ZSTR_INCLUDE_DIR)

# Setup the imported target
if (NOT TARGET ZSTR::ZSTR)
  add_library(ZSTR::ZSTR INTERFACE IMPORTED)
endif (NOT TARGET ZSTR::ZSTR)

# Set the include directories for the target
set_property(TARGET ZSTR::ZSTR APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ZSTR_INCLUDE_DIRS})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE ZSTR_INCLUDE_DIRS)
