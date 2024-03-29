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
# CONDUIT-specific configuration options explicitly exposed
option(LBANN_SB_FWD_Conduit_ENABLE_MPI
  "Enable MPI support in CONDUIT."
  ON)
option(LBANN_SB_FWD_Conduit_ENABLE_PYTHON
  "Enable CONDUIT Python module support."
  OFF)
option(LBANN_SB_FWD_Conduit_ENABLE_TESTS
  "Build CONDUIT unit tests."
  OFF)
option(LBANN_SB_FWD_Conduit_ENABLE_FORTRAN
  "Build CONDUIT with fortran support"
  OFF)

set(lbann_sb_ftn_lang)
if (LBANN_SB_FWD_Conduit_ENABLE_FORTRAN)
  set(lbann_sb_ftn_lang Fortran)
endif ()

# Conduit is "cute" about finding HDF5. It's not a CMake option() --
# you opt in by setting HDF5_DIR explicitly. So let's do that.
if (TARGET HDF5 AND NOT LBANN_SB_FWD_Conduit_HDF5_DIR)
  set(LBANN_SB_FWD_Conduit_HDF5_DIR ${HDF5_DIR})
endif ()

lbann_sb_add_cmake_extern_pkg(
  NAME Conduit
  LANGUAGES C CXX ${lbann_sb_ftn_lang}
  GITHUB_URL llnl/conduit.git
  GIT_TAG "develop"
  SOURCE_SUBDIR src
  DEPENDS_ON HDF5
)

set(Conduit_DIR ${LBANN_SB_Conduit_PREFIX}
  CACHE INTERNAL "The install prefix of Conduit.")
