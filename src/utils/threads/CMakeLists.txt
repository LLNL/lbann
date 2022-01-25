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
# Test for non-portable POSIX features
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
set(_PTHREAD_SETAFFINITY_NP_TEST_CODE
  "#include <pthread.h>
int main(int, char* argv[]) {
  pthread_t thd = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  return pthread_setaffinity_np(thd, sizeof(cpu_set_t), &cpuset);
}")
check_cxx_source_compiles(
  "${_PTHREAD_SETAFFINITY_NP_TEST_CODE}" LBANN_HAS_PTHREAD_SETAFFINITY_NP)
set(_PTHREAD_GETAFFINITY_NP_TEST_CODE
  "#include <pthread.h>
int main(int, char* argv[]) {
  pthread_t thd = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  return pthread_getaffinity_np(thd, sizeof(cpu_set_t), &cpuset);
}")
check_cxx_source_compiles(
  "${_PTHREAD_GETAFFINITY_NP_TEST_CODE}" LBANN_HAS_PTHREAD_GETAFFINITY_NP)
set(CMAKE_REQUIRED_LIBRARIES)

if (LBANN_HAS_PTHREAD_SETAFFINITY_NP AND LBANN_HAS_PTHREAD_GETAFFINITY_NP)
  set(LBANN_HAS_PTHREAD_AFFINITY_SUPPORT TRUE CACHE INTERNAL
    "LBANN has pthread affinity support")
endif ()

# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  thread_pool.cpp
  thread_utils.cpp
  thread_topology.cpp
)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
