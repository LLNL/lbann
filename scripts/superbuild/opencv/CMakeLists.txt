################################################################################
## Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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
# OpenCV will default to using JPEG-TURBO if it's being built with the
# superbuild at the same time.
if (TARGET JPEG-TURBO)
  set(LBANN_SB_FWD_OpenCV_WITH_LIBJPEG_TURBO ON)
  set(LBANN_SB_FWD_OpenCV_BUILD_JPEG OFF)
  set(LBANN_SB_FWD_OpenCV_CMAKE_PREFIX_PATH
    ${CMAKE_PREFIX_PATH}
    "${LBANN_SB_JPEG-TURBO_PREFIX}")
endif ()

# Add all the OpenCV options; this is just for the GUI, of course
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(AddOpenCVOptions)

lbann_sb_add_cmake_extern_pkg(
  NAME OpenCV
  LANGUAGES C CXX Fortran # WHY FORTRAN???
  GITHUB_URL opencv/opencv
  GIT_TAG "4.1.0"
  DEPENDS_ON JPEG-TURBO
)

set(OpenCV_DIR ${LBANN_SB_OpenCV_PREFIX}
  CACHE INTERNAL "The install prefix of OpenCV.")
