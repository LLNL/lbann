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
# Match the CNPY default
option(LBANN_SB_FWD_CNPY_ENABLE_STATIC "Enable the CNPY static linkage." ON)

lbann_sb_add_cmake_extern_pkg(
  NAME CNPY
  LANGUAGES C CXX
  GITHUB_URL rogersce/cnpy.git
  GIT_TAG "4e8810b1a8637695171ed346ce68f6984e585ef4"
  EXTRA_CMAKE_ARGS
  -D CMAKE_MACOSX_RPATH=ON)

set(CNPY_DIR ${LBANN_SB_CNPY_PREFIX}
  CACHE INTERNAL "The install prefix of CNPY.")
