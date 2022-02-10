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

# NOTE: If using AppleClang on OSX, you'll probably want to set
#
#   OpenMP_CXX_FLAGS
#
# on the command line. An example, using the latest Homebrew
# installation, might be:
#
#   -DOpenMP_CXX_FLAGS="-fopenmp -I/usr/local/include/libiomp/ -L/usr/local/lib/"

if (NOT OpenMP_CXX_FOUND)
  find_package(OpenMP REQUIRED CXX)
endif ()

if (NOT TARGET OpenMP::OpenMP_CXX)
  add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)

  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
    INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)

  # Propagate to the link flags
  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
    INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS})
  # The imported target will be defined in the same version as CMake
  # introduced the "OpenMP_<lang>_LIBRARIES" variable. Thus we don't
  # provide a contingency for them here.

else ()
  get_target_property(_OMP_FLAGS OpenMP::OpenMP_CXX INTERFACE_COMPILE_OPTIONS)

  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
    INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:${_OMP_FLAGS}>)
endif ()
