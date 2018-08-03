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
#include "lbann/utils/random.hpp"
#include "lbann/io/file_io.hpp"

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

bool save_rng_to_checkpoint_shared(persist& p, const lbann_comm* comm) {
  std::string dirname = std::string(p.m_checkpoint_dir) + "/rng_state";
  makedir(dirname.c_str());
  std::string rng_name;

  rng_name = dirname + "/rng_seq_generator";
  std::ofstream rng_seq(rng_name);
  rng_seq << ::data_seq_generator;
#ifdef LBANN_SET_EL_RNG
  rng_name = dirname + "/EL_generator";
  std::ofstream rng_EL(rng_name);
  rng_EL << El::Generator();
#endif

  std::string rank_in_world;
  if (comm == nullptr) {
    rank_in_world = std::to_string(El::mpi::Rank(El::mpi::COMM_WORLD));
  } else {
    rank_in_world = std::to_string(comm->get_rank_in_world());

  }
#ifdef _OPENMP
  #pragma omp parallel private(rng_name)
  {
    rng_name = dirname + "/rng_generator_" + rank_in_world + "_" + std::to_string(omp_get_thread_num());
    std::ofstream rng(rng_name);
    rng << ::generator;

    rng_name = dirname + "/rng_fast_generator_" + rank_in_world + "_" + std::to_string(omp_get_thread_num());
    std::ofstream rng_fast(rng_name);
    rng_fast << ::fast_generator;
  }
#else
    rng_name = dirname + "/rng_generator_" + rank_in_world;
    std::ofstream rng(rng_name);
    rng << ::generator;

    rng_name = dirname + "/rng_fast_generator_" + rank_in_world;
    std::ofstream rng_fast(rng_name);
    rng_fast << ::fast_generator;
#endif

   return true;
}

bool load_rng_from_checkpoint_shared(persist& p, const lbann_comm* comm) {

  std::string dirname = std::string(p.m_checkpoint_dir) + "/rng_state";
  std::string rng_name;

  rng_name = dirname + "/rng_seq_generator";
  std::ifstream rng_seq(rng_name);
  rng_seq >> ::data_seq_generator;
#ifdef LBANN_SET_EL_RNG
  rng_name = dirname + "/EL_generator";
  std::ifstream rng_EL(rng_name);
  rng_EL >> El::Generator();
#endif

  std::string rank_in_world;
  if (comm == nullptr) {
    rank_in_world = std::to_string(El::mpi::Rank(El::mpi::COMM_WORLD));
  } else {
    rank_in_world = std::to_string(comm->get_rank_in_world());
  }

 #ifdef _OPENMP
  #pragma omp parallel private(rng_name)
  {
    rng_name = dirname + "/rng_generator_" + rank_in_world + "_" + std::to_string(omp_get_thread_num());
    std::ifstream rng(rng_name);
    rng >> ::generator;

    rng_name = dirname + "/rng_fast_generator_" + rank_in_world + "_" + std::to_string(omp_get_thread_num());
    std::ifstream rng_fast(rng_name);
    rng_fast >> ::fast_generator;
   }
#else
    rng_name = dirname + "/rng_generator_" + rank_in_world;
    std::ifstream rng(rng_name);
    rng >> ::generator;

    rng_name = dirname + "/rng_fast_generator_" + rank_in_world;
    std::ifstream rng_fast(rng_name);
    rng_fast >> ::fast_generator;
   }
#endif
  return true;
}

void init_random(int seed, lbann_comm *comm) {
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
      get_generator().seed((rand_val << 8) | omp_get_thread_num());
      get_fast_generator().seed((rand_val << 8) | omp_get_thread_num());
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

void gaussian_fill(AbsDistMat& mat, IntType m, IntType n, DataType mean,
                   DataType stddev) {
#ifndef LBANN_DETERMINISTIC
  El::Gaussian(mat, m, n, mean, stddev);
#else
  gaussian_fill_procdet(mat, m, n, mean, stddev);
#endif  // LBANN_PARALLEL_DETERMINISTIC
}

void bernoulli_fill(AbsDistMat& mat, IntType m, IntType n, double p) {
#ifndef LBANN_DETERMINISTIC
  El::Bernoulli(mat, m, n, p);
#else
  bernoulli_fill_procdet(mat, m, n, p);
#endif  // LBANN_PARALLEL_DETERMINISTIC
}

void uniform_fill(AbsDistMat& mat, IntType m, IntType n, DataType center,
                  DataType radius) {
#ifndef LBANN_DETERMINISTIC
  El::Uniform(mat, m, n, center, radius);
#else
  uniform_fill_procdet(mat, m, n, center, radius);
#endif  // LBANN_PARALLEL_DETERMINISTIC
}

void gaussian_fill_procdet(AbsDistMat& mat, IntType m, IntType n, DataType mean,
                           DataType stddev) {
  CircMat<El::Device::CPU> vals(m, n, mat.Grid(), 0);
  if (vals.Participating()) {
    auto& local_vals = vals.Matrix();
    auto& gen = get_generator();
    std::normal_distribution<DataType> dist(mean, stddev);
    for (IntType col = 0; col < local_vals.Width(); ++col) {
      for (IntType row = 0; row < local_vals.Height(); ++row) {
        local_vals(row, col) = dist(gen);
      }
    }
  }
  El::Copy(vals, mat);
}

void bernoulli_fill_procdet(AbsDistMat& mat, IntType m, IntType n, double p) {
  CircMat<El::Device::CPU> vals(m, n, mat.Grid(), 0);
  if (vals.Participating()) {
    auto& local_vals = vals.Matrix();
    auto& gen = get_generator();
    std::bernoulli_distribution dist(p);
    for (IntType col = 0; col < local_vals.Width(); ++col) {
      for (IntType row = 0; row < local_vals.Height(); ++row) {
        local_vals(row, col) = dist(gen);
      }
    }
  }
  El::Copy(vals, mat);
}

void uniform_fill_procdet(AbsDistMat& mat, IntType m, IntType n, DataType center,
                          DataType radius) {
  CircMat<El::Device::CPU> vals(m, n, mat.Grid(), 0);
  if (vals.Participating()) {
    auto& local_vals = vals.Matrix();
    auto& gen = get_generator();
    std::uniform_real_distribution<DataType> dist(center - radius,
                                                  center + radius);
    for (IntType col = 0; col < local_vals.Width(); ++col) {
      for (IntType row = 0; row < local_vals.Height(); ++row) {
        local_vals(row, col) = dist(gen);
      }
    }
  }
  El::Copy(vals, mat);
}

}  // namespace lbann
