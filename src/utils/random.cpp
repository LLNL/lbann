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
#include "lbann/utils/random.hpp"
#include "lbann/io/file_io.hpp"
#include <thread>

namespace {
#ifdef __ICC
lbann::rng_gen generator;
#pragma omp threadprivate(generator)

lbann::fast_rng_gen fast_generator;
#pragma omp threadprivate(fast_generator)
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
#endif

thread_local lbann::rng_gen data_seq_generator;
thread_local bool data_seq_generator_inited = false;
int data_seq_generator_seed_base = 0;

thread_local lbann::rng_gen io_generator;
thread_local bool io_generator_inited = false;
int io_generator_seed_base = 0;

thread_local lbann::fast_rng_gen fast_io_generator;
thread_local bool fast_io_generator_inited = false;
int fast_io_generator_seed_base = 0;
}

namespace lbann {

rng_gen& get_generator() {
  return ::generator;
}

fast_rng_gen& get_fast_generator() {
  return ::fast_generator;
}

rng_gen& get_data_seq_generator() {
  if (!::data_seq_generator_inited) {
    ::data_seq_generator.seed(::data_seq_generator_seed_base);
    ::data_seq_generator_inited = true;
  }
  return ::data_seq_generator;
}

rng_gen& get_io_generator() {
  if (!::io_generator_inited) {
    std::hash<std::thread::id> h;
    ::io_generator.seed((::io_generator_seed_base << 8) |
                        h(std::this_thread::get_id()));
    ::io_generator_inited = true;
  }
  return ::io_generator;
}

fast_rng_gen& get_fast_io_generator() {
  if (!::fast_io_generator_inited) {
    std::hash<std::thread::id> h;
    ::fast_io_generator.seed((::fast_io_generator_seed_base << 8) |
                             h(std::this_thread::get_id()));
    ::fast_io_generator_inited = true;
  }
  return ::fast_io_generator;
}

bool save_rng_to_checkpoint_shared(persist& p, const lbann_comm* comm) {
  std::string dirname = std::string(p.m_checkpoint_dir) + "/rng_state";
  makedir(dirname.c_str());
  std::string rng_name;

  /// @todo - Note that the RNG with thread local data is not correct
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

  /// @todo - Note that the RNG with thread local data is not correct
  rng_name = dirname + "/rng_io_generator_" + rank_in_world;
  std::ofstream rng_io(rng_name);
  rng_io << ::io_generator;

  /// @todo - Note that the RNG with thread local data is not correct
  rng_name = dirname + "/rng_fast_io_generator_" + rank_in_world;
  std::ofstream rng_fast_io(rng_name);
  rng_fast_io << ::fast_io_generator;

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

  /// @todo - Note that the RNG with thread local data is not correct
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

  /// @todo - Note that the RNG with thread local data is not correct
  rng_name = dirname + "/rng_io_generator_" + rank_in_world;
  std::ifstream rng_io(rng_name);
  rng_io >> ::io_generator;

  /// @todo - Note that the RNG with thread local data is not correct
  rng_name = dirname + "/rng_fast_io_generator_" + rank_in_world;
  std::ifstream rng_fast_io(rng_name);
  rng_fast_io >> ::fast_io_generator;

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
      El::Generator().seed(seed ^ comm->get_rank_in_trainer());
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

  init_io_random(seed);
}

void init_data_seq_random(int seed) {
  if (seed == -1) {
    // Seed with a random value.
    std::random_device rd;
    seed = rd();
  }

  ::data_seq_generator_seed_base = seed;
  /// Reset the init flag so that generator will reinitialize
  ::data_seq_generator_inited = false;
}

void init_io_random(int seed) {
  if (seed == -1) {
    // Seed with a random value.
    std::random_device rd;
    seed = rd();
  }

  ::io_generator_seed_base = seed;
  /// Reset the init flag so that generator will reinitialize
  ::io_generator_inited = false;

  ::fast_io_generator_seed_base = seed;
  /// Reset the init flag so that generator will reinitialize
  ::fast_io_generator_inited = false;
}

void gaussian_fill(AbsDistMat& mat, El::Int m, El::Int n, DataType mean,
                   DataType stddev) {
#ifndef LBANN_DETERMINISTIC
  El::Gaussian(mat, m, n, mean, stddev);
#else
  gaussian_fill_procdet(mat, m, n, mean, stddev);
#endif  // LBANN_PARALLEL_DETERMINISTIC
}

void bernoulli_fill(AbsDistMat& mat, El::Int m, El::Int n, double p) {
#ifndef LBANN_DETERMINISTIC
  El::Bernoulli(mat, m, n, p);
#else
  bernoulli_fill_procdet(mat, m, n, p);
#endif  // LBANN_PARALLEL_DETERMINISTIC
}

void uniform_fill(AbsDistMat& mat, El::Int m, El::Int n, DataType center,
                  DataType radius) {
#ifndef LBANN_DETERMINISTIC
  El::Uniform(mat, m, n, center, radius);
#else
  uniform_fill_procdet(mat, m, n, center, radius);
#endif  // LBANN_PARALLEL_DETERMINISTIC
}

void gaussian_fill_procdet(AbsDistMat& mat, El::Int m, El::Int n, DataType mean,
                           DataType stddev) {
  CircMat<El::Device::CPU> vals(m, n, mat.Grid(), 0);
  if (vals.Participating()) {
    auto& local_vals = vals.Matrix();
    auto& gen = get_generator();
    std::normal_distribution<DataType> dist(mean, stddev);
    for (El::Int col = 0; col < local_vals.Width(); ++col) {
      for (El::Int row = 0; row < local_vals.Height(); ++row) {
        local_vals(row, col) = dist(gen);
      }
    }
  }
  El::Copy(vals, mat);
}

void bernoulli_fill_procdet(AbsDistMat& mat, El::Int m, El::Int n, double p) {
  CircMat<El::Device::CPU> vals(m, n, mat.Grid(), 0);
  if (vals.Participating()) {
    auto& local_vals = vals.Matrix();
    auto& gen = get_generator();
    std::bernoulli_distribution dist(p);
    for (El::Int col = 0; col < local_vals.Width(); ++col) {
      for (El::Int row = 0; row < local_vals.Height(); ++row) {
        local_vals(row, col) = dist(gen);
      }
    }
  }
  El::Copy(vals, mat);
}

void uniform_fill_procdet(AbsDistMat& mat, El::Int m, El::Int n, DataType center,
                          DataType radius) {
  CircMat<El::Device::CPU> vals(m, n, mat.Grid(), 0);
  if (vals.Participating()) {
    auto& local_vals = vals.Matrix();
    auto& gen = get_generator();
    std::uniform_real_distribution<DataType> dist(center - radius,
                                                  center + radius);
    for (El::Int col = 0; col < local_vals.Width(); ++col) {
      for (El::Int row = 0; row < local_vals.Height(); ++row) {
        local_vals(row, col) = dist(gen);
      }
    }
  }
  El::Copy(vals, mat);
}

}  // namespace lbann
