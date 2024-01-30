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

# This prefers Catch2 v3
lbann_sb_add_cmake_extern_pkg(
  NAME Catch2
  LANGUAGES CXX
  GITHUB_URL catchorg/catch2.git
  GIT_TAG "devel"
  EXTRA_CMAKE_ARGS
  -DCATCH_BUILD_TESTING=OFF
  -DCATCH_BUILD_EXAMPLES=OFF
  -DCATCH_ENABLE_WERROR=OFF
  )

set(CATCH2_DIR ${LBANN_SB_Catch2_PREFIX}
  CACHE INTERNAL "The install prefix of Catch2.")
