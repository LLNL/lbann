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

#include <omp.h>
#include "lbann/utils/lbann_random.hpp"

namespace {
// Random number generator, file-visible only.
// Defined like this to work around a GCC problem with threadprivate objects:
// https://stackoverflow.com/questions/23552077/how-to-define-a-object-or-struct-as-threadprivate-in-openmp/
extern lbann::rng_gen generator;
#pragma omp threadprivate(generator)
lbann::rng_gen generator;

extern lbann::fast_rng_gen fast_generator;
#pragma omp threadprivate(fast_generator)
lbann::fast_rng_gen fast_generator;
}

namespace lbann {

rng_gen& get_generator() {
  return ::generator;
}

fast_rng_gen& get_fast_generator() {
  return ::fast_generator;
}

void init_random(int seed, lbann_comm* comm) {
  if (seed != -1) {
    // Seed every OpenMP thread, if present.
    // Note: Threadprivate OMP variables don't work with dynamic threads.
#ifdef _OPENMP
    #pragma omp parallel
    {
      get_generator().seed((seed << 8) | (omp_get_thread_num() & 0xff));
      get_fast_generator().seed((seed << 8) | (omp_get_thread_num() & 0xff));
    }
#else
    get_generator().seed(seed);
    get_fast_generator().seed(seed);
#endif
#ifdef LBANN_SET_EL_RNG
    if (comm != nullptr) {
      El::Generator().seed(seed ^ comm->get_rank_in_model());
    } else {
      El::Generator().seed(seed ^ El::mpi::Rank(El::mpi::COMM_WORLD));
    }
#endif
  } else {
    // Seed with a random value.
    std::random_device rd;
    unsigned rand_val = rd();
#ifdef _OPENMP
    #pragma omp parallel
    {
      get_generator().seed((rand_val << 8) | (omp_get_thread_num() & 0xff));
      get_fast_generator().seed((rand_val << 8) | (omp_get_thread_num() & 0xff));
    }
#else
    get_generator().seed(rand_val);
    get_fast_generator().seed(rand_val);
#endif
#ifdef LBANN_SET_EL_RNG
    El::Generator().seed(rand_val);
#endif
  }
}

void gaussian_fill(ElMat& mat, El::Int m, El::Int n, DataType mean,
                   DataType stddev) {
#ifdef LBANN_PARALLEL_RANDOM_MATRICES
  El::Gaussian(mat, m, n, mean, stddev);
#else
  gaussian_fill_procdet(mat, m, n, mean, stddev);
#endif  // LBANN_PARALLEL_RANDOM_MATRICES
}

void bernoulli_fill(ElMat& mat, El::Int m, El::Int n, double p) {
#ifdef LBANN_PARALLEL_RANDOM_MATRICES
  El::Bernoulli(mat, m, n, p);
#else
  bernoulli_fill_procdet(mat, m, n, p);
#endif  // LBANN_PARALLEL_RANDOM_MATRICES  
}

void uniform_fill(ElMat& mat, El::Int m, El::Int n, DataType center,
                   DataType radius) {
#ifdef LBANN_PARALLEL_RANDOM_MATRICES
  El::Uniform(mat, m, n, center, radius);
#else
  uniform_fill_procdet(mat, m, n, center, radius);
#endif  // LBANN_PARALLEL_RANDOM_MATRICES
}

void gaussian_fill_procdet(ElMat& mat, El::Int m, El::Int n, DataType mean,
                           DataType stddev) {
  Zeros(mat, m, n);
  if (mat.Grid().Rank() == 0) {
    mat.Reserve(n * m);
    auto& gen = get_generator();
    std::normal_distribution<DataType> dist(mean, stddev);
    for (El::Int col = 0; col < n; ++col) {
      for (El::Int row = 0; row < m; ++row) {
        mat.QueueUpdate(row, col, dist(gen));
      }
    }
  }
  mat.ProcessQueues();
}

void bernoulli_fill_procdet(ElMat& mat, El::Int m, El::Int n, double p) {
  Zeros(mat, m, n);
  if (mat.Grid().Rank() == 0) {
    mat.Reserve(m * n);
    auto& gen = get_generator();
    std::bernoulli_distribution dist(p);
    for (El::Int col = 0; col < n; ++col) {
      for (El::Int row = 0; row < m; ++row) {
        mat.QueueUpdate(row, col, dist(gen) ? 1.0f : 0.0f);
      }
    }
  }
  mat.ProcessQueues();
}

void uniform_fill_procdet(ElMat& mat, El::Int m, El::Int n, DataType center,
                          DataType radius) {
  Zeros(mat, m, n);
  if (mat.Grid().Rank() == 0) {
    mat.Reserve(n * m);
    auto& gen = get_generator();
    std::uniform_real_distribution<DataType> dist(center - radius,
                                                  center + radius);
    for (El::Int col = 0; col < n; ++col) {
      for (El::Int row = 0; row < m; ++row) {
        mat.QueueUpdate(row, col, dist(gen));
      }
    }
  }
  mat.ProcessQueues();
}

}  // namespace lbann

