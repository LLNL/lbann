////////////////////////////////////////////////////////////////////////////////xecu
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
//
// lbann_base .cpp - Basic definitions, functions
////////////////////////////////////////////////////////////////////////////////

#include <thread>
#include <omp.h>
#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"

namespace lbann {

void initialize(lbann_comm* comm) {
#ifdef _OPENMP
  // Initialize the default number of threads to use for parallel regions.
  // Note the num_threads directive can override this if specifically set.
  // Further, if the OMP_NUM_THREADS environment variable is set, we don't
  // change it.
  if (getenv("OMP_NUM_THREADS") == NULL) {
    const int threads_per_rank = std::thread::hardware_concurrency() /
      comm->get_procs_per_node();
    omp_set_num_threads(threads_per_rank);
  }
#endif  // _OPENMP
}

void finalize() {

}

}  // namespace lbann
