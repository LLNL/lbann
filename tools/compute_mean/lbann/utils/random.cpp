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

#include <omp.h>
#include "random.hpp"

namespace {
#ifdef __ICC
lbann::rng_gen generator;
#pragma omp threadprivate(generator)

lbann::fast_rng_gen fast_generator;
#pragma omp threadprivate(fast_generator)

lbann::rng_gen data_seq_generator;
#pragma omp threadprivate(data_seq_generator)
#else
// Random number generator, file-visible only.
// Defined like this to work around a GCC problem with threadprivate objects:
// https://stackoverflow.com/questions/23552077/how-to-define-a-object-or-struct-as-threadprivate-in-openmp/
extern lbann::rng_gen generator;
#pragma omp threadprivate(generator)
lbann::rng_gen generator;

extern lbann::fast_rng_gen fast_generator;
#pragma omp threadprivate(fast_generator)
lbann::fast_rng_gen fast_generator;

extern lbann::rng_gen data_seq_generator;
#pragma omp threadprivate(data_seq_generator)
lbann::rng_gen data_seq_generator;
#endif
}

namespace lbann {

rng_gen& get_generator() {
  return ::generator;
}

fast_rng_gen& get_fast_generator() {
  return ::fast_generator;
}

rng_gen& get_data_seq_generator() {
  return ::data_seq_generator;
}

void init_random(int seed) {
  if (seed != -1) {
    // Seed every OpenMP thread, if present.
    // Note: Threadprivate OMP variables don't work with dynamic threads.
#ifdef _OPENMP
    #pragma omp parallel
    {
      get_generator().seed((seed << 8) | omp_get_thread_num());
      get_fast_generator().seed((seed << 8) | omp_get_thread_num());
    }
#else
    get_generator().seed(seed);
    get_fast_generator().seed(seed);
#endif
  } else {
    // Seed with a random value.
    std::random_device rd;
    unsigned rand_val = rd();
#ifdef _OPENMP
    #pragma omp parallel
    {
      get_generator().seed((rand_val << 8) | omp_get_thread_num());
      get_fast_generator().seed((rand_val << 8) | omp_get_thread_num());
    }
#else
    get_generator().seed(rand_val);
    get_fast_generator().seed(rand_val);
#endif
  }
}

void init_data_seq_random(int seed) {
  if (seed == -1) {
    // Seed with a random value.
    std::random_device rd;
    seed = rd();
  }

  // Seed every OpenMP thread, if present.
  // Note: Threadprivate OMP variables don't work with dynamic threads.
#ifdef _OPENMP
  #pragma omp parallel
  {
    get_data_seq_generator().seed(seed);
  }
#else
  get_data_seq_generator().seed(seed);
#endif
}

}  // namespace lbann
