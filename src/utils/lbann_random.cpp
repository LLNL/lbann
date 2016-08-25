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

#include "lbann/utils/lbann_random.hpp"

namespace {
// Random number generator, file-visible only.
lbann::rng_gen generator;
}

namespace lbann {

rng_gen& get_generator() {
  return ::generator;
}

void init_random(int seed) {
  if (seed != -1) {
    get_generator().seed(seed);
#ifdef LBANN_SET_EL_RNG
    El::Generator().seed(seed);
#endif
  } else {
    // Seed with a random value.
    std::random_device rd;
    unsigned rand_val = rd();
    get_generator().seed(rand_val);
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
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < m; ++row) {
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
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < m; ++row) {
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
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < m; ++row) {
        mat.QueueUpdate(row, col, dist(gen));
      }
    }
  }
  mat.ProcessQueues();
}

}  // namespace lbann

