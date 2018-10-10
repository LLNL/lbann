////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#define OMP_PARALLEL _Pragma("omp parallel for")
#define OMP_CRITICAL _Pragma("omp critical")

#if defined(LBANN_NO_OMP_FOR_DATA_READERS)
  #pragma message "Disable OpenMP parallelism for data fetch loops"
  #define LBANN_DATA_FETCH_OMP_FOR for
  #define LBANN_OMP_THREAD_NUM 0
  #define LBANN_DATA_FETCH_OMP_CRITICAL
#else
  #define LBANN_DATA_FETCH_OMP_FOR OMP_PARALLEL for
  #define LBANN_OMP_THREAD_NUM omp_get_thread_num()
  #define LBANN_DATA_FETCH_OMP_CRITICAL OMP_CRITICAL
#endif

#endif // LBANN_OMP_PRAGMA_HPP
