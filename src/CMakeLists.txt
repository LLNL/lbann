################################################################################
## Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Elemental_extensions.cpp
  base.cpp
  comm.cpp
  )

# Add the subdirectories
add_subdirectory(callbacks)
add_subdirectory(data_ingestion)
add_subdirectory(io)
add_subdirectory(layers)
add_subdirectory(metrics)
add_subdirectory(models)
add_subdirectory(objective_functions)
add_subdirectory(operators)
add_subdirectory(optimizers)
add_subdirectory(proto)
add_subdirectory(trainers)
add_subdirectory(execution_algorithms)
add_subdirectory(transforms)
add_subdirectory(utils)
add_subdirectory(weights)

if (TARGET LbannProto)
  set(LBANN_HAS_LBANN_PROTO TRUE PARENT_SCOPE)
endif ()

# Propagate the files up the tree
set(LBANN_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
set(LBANN_SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(LBANN_GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
