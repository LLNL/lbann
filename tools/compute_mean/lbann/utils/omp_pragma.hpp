////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OMP_PRAGMA_HPP
#define LBANN_OMP_PRAGMA_HPP

#include "lbann_config.hpp"
#include <omp.h>

/// Allow OpenMP parallel for loops to be replaced with taskloop constructs
/// Requires OpenMP 5.0 support for taskloop reduction clauses
#if defined(LBANN_HAVE_OMP_TASKLOOP)
  #pragma message "Using OpenMP taskloops instead of parallel for loops"
  #define LBANN_OMP_PARALLEL_FOR_HELPER(arg) #arg
  #define LBANN_OMP_PARALLEL_FOR_TEXT(arg) LBANN_OMP_PARALLEL_FOR_HELPER(omp taskloop default(shared) num_tasks(omp_get_num_threads()) arg)
  #define LBANN_OMP_PARALLEL_FOR_ARGS(arg) _Pragma(LBANN_OMP_PARALLEL_FOR_TEXT(arg))

  #define LBANN_OMP_PARALLEL_FOR _Pragma("omp taskloop default(shared) num_tasks(omp_get_num_threads())")
  #define LBANN_OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp taskloop collapse(2) default(shared) num_tasks(omp_get_num_threads())")
  #define LBANN_OMP_PARALLEL_FOR_COLLAPSE3 _Pragma("omp taskloop collapse(3) default(shared) num_tasks(omp_get_num_threads())")
  #define LBANN_OMP_PARALLEL_FOR_COLLAPSE4 _Pragma("omp taskloop collapse(4) default(shared) num_tasks(omp_get_num_threads())")
  #define LBANN_OMP_PARALLEL_FOR_COLLAPSE5 _Pragma("omp taskloop collapse(5) default(shared) num_tasks(omp_get_num_threads())")
#else
  #define LBANN_OMP_PARALLEL_FOR_HELPER(arg) #arg
  #define LBANN_OMP_PARALLEL_FOR_TEXT(arg) LBANN_OMP_PARALLEL_FOR_HELPER(omp parallel for arg)
  #define LBANN_OMP_PARALLEL_FOR_ARGS(arg) _Pragma(LBANN_OMP_PARALLEL_FOR_TEXT(arg))

  #define LBANN_OMP_PARALLEL_FOR _Pragma("omp parallel for")
  #define LBANN_OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2)")
  #define LBANN_OMP_PARALLEL_FOR_COLLAPSE3 _Pragma("omp parallel for collapse(3)")
  #define LBANN_OMP_PARALLEL_FOR_COLLAPSE4 _Pragma("omp parallel for collapse(4)")
  #define LBANN_OMP_PARALLEL_FOR_COLLAPSE5 _Pragma("omp parallel for collapse(5)")
#endif

#define LBANN_OMP_PARALLEL_HELPER(arg) #arg
#define LBANN_OMP_PARALLEL_TEXT(arg) LBANN_OMP_PARALLEL_HELPER(omp parallel arg)
#define LBANN_OMP_PARALLEL_ARGS(arg) _Pragma(LBANN_OMP_PARALLEL_TEXT(arg))

#define LBANN_OMP_PARALLEL _Pragma("omp parallel")
#define OMP_CRITICAL _Pragma("omp critical")

#endif // LBANN_OMP_PRAGMA_HPP
