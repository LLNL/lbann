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
#define LBANN_RANDOM_INSTANTIATE
#include "lbann/utils/random.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/utils/hash.hpp"
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
    ::io_generator.seed(hash_combine(::io_generator_seed_base,
                                     std::this_thread::get_id()));
    ::io_generator_inited = true;
  }
  return ::io_generator;
}

fast_rng_gen& get_fast_io_generator() {
  if (!::fast_io_generator_inited) {
    ::fast_io_generator.seed(hash_combine(::fast_io_generator_seed_base,
                                          std::this_thread::get_id()));
    ::fast_io_generator_inited = true;
  }
  return ::fast_io_generator;
}

bool save_rng_to_checkpoint(persist& p, lbann_comm* comm, bool is_distributed) {
  std::string dirname = std::string(p.m_checkpoint_dir) + "/rng_state";
  std::string rank_in_trainer;
  std::string rng_name;

  if (comm == nullptr) {
    rank_in_trainer = std::to_string(El::mpi::Rank(El::mpi::COMM_WORLD));
    makedir(dirname.c_str());
  } else {
    rank_in_trainer = std::to_string(comm->get_rank_in_trainer());
    if (comm->am_trainer_master() || is_distributed) {
      makedir(dirname.c_str());
    }
    comm->trainer_barrier();
  }

  if (comm == nullptr || comm->am_trainer_master() || is_distributed) {
    /// @todo - Note that the RNG with thread local data is not correct
    rng_name = dirname + "/rng_seq_generator";
    std::ofstream rng_seq(rng_name);
    if(!rng_seq) { LBANN_ERROR("Failed to open ", rng_name); }
    rng_seq << ::data_seq_generator;
    rng_seq.close();

#ifdef LBANN_SET_EL_RNG
    rng_name = dirname + "/EL_generator";
    std::ofstream rng_EL(rng_name);
    if(!rng_EL) { LBANN_ERROR("Failed to open ", rng_name); }
    rng_EL << El::Generator();
    rng_EL.close();
#endif
  }

  /// @todo - Note that the RNG with thread local data is not correct
  rng_name = dirname + "/rng_io_generator_" + rank_in_trainer;
  std::ofstream rng_io(rng_name);
  if(!rng_io) { LBANN_ERROR("Failed to open ", rng_name); }
  rng_io << ::io_generator;
  rng_io.close();

  /// @todo - Note that the RNG with thread local data is not correct
  rng_name = dirname + "/rng_fast_io_generator_" + rank_in_trainer;
  std::ofstream rng_fast_io(rng_name);
  if(!rng_fast_io) { LBANN_ERROR("Failed to open ", rng_name); }
  rng_fast_io << ::fast_io_generator;
  rng_fast_io.close();

#ifdef _OPENMP
  #pragma omp parallel private(rng_name)
  {
    rng_name = dirname + "/rng_generator_" + rank_in_trainer + "_"
             + std::to_string(omp_get_thread_num());
    std::ofstream rng(rng_name);
    if(!rng) { LBANN_ERROR("Failed to open ", rng_name); }
    rng << ::generator;
    rng.close();

    rng_name = dirname + "/rng_fast_generator_" + rank_in_trainer + "_"
             + std::to_string(omp_get_thread_num());
    std::ofstream rng_fast(rng_name);
    if(!rng_fast) { LBANN_ERROR("Failed to open ", rng_name); }
    rng_fast << ::fast_generator;
    rng_fast.close();
  }
#else
    rng_name = dirname + "/rng_generator_" + rank_in_trainer;
    std::ofstream rng(rng_name);
    if(!rng) { LBANN_ERROR("Failed to open ", rng_name); }
    rng << ::generator;
    rng.close();

    rng_name = dirname + "/rng_fast_generator_" + rank_in_trainer;
    std::ofstream rng_fast(rng_name);
    if(!rng_fast) { LBANN_ERROR("Failed to open ", rng_name); }
    rng_fast << ::fast_generator;
    rng_fast.close();
#endif

   return true;
}

bool save_rng_to_checkpoint_shared(persist& p, lbann_comm* comm) {
  return save_rng_to_checkpoint(p, comm, false);
}

bool save_rng_to_checkpoint_distributed(persist& p, lbann_comm* comm) {
  return save_rng_to_checkpoint(p, comm, true);
}

bool load_rng_from_checkpoint(persist& p, const lbann_comm* comm) {

  std::string dirname = std::string(p.m_checkpoint_dir) + "/rng_state";
  std::string rng_name;

  /// @todo - Note that the RNG with thread local data is not correct
  rng_name = dirname + "/rng_seq_generator";
  std::ifstream rng_seq(rng_name);
  if(!rng_seq) { LBANN_ERROR("Failed to open ", rng_name); }
  rng_seq >> ::data_seq_generator;

#ifdef LBANN_SET_EL_RNG
  rng_name = dirname + "/EL_generator";
  std::ifstream rng_EL(rng_name);
  if(!rng_EL) { LBANN_ERROR("Failed to open ", rng_name); }
  rng_EL >> El::Generator();
#endif

  std::string rank_in_trainer;
  if (comm == nullptr) {
    rank_in_trainer = std::to_string(El::mpi::Rank(El::mpi::COMM_WORLD));
  } else {
    rank_in_trainer = std::to_string(comm->get_rank_in_trainer());
  }

  /// @todo - Note that the RNG with thread local data is not correct
  rng_name = dirname + "/rng_io_generator_" + rank_in_trainer;
  std::ifstream rng_io(rng_name);
  if(!rng_io) { LBANN_ERROR("Failed to open ", rng_name); }
  rng_io >> ::io_generator;

  /// @todo - Note that the RNG with thread local data is not correct
  rng_name = dirname + "/rng_fast_io_generator_" + rank_in_trainer;
  std::ifstream rng_fast_io(rng_name);
  if(!rng_fast_io) { LBANN_ERROR("Failed to open ", rng_name); }
  rng_fast_io >> ::fast_io_generator;

#ifdef _OPENMP
  #pragma omp parallel private(rng_name)
  {
    rng_name = dirname + "/rng_generator_" + rank_in_trainer + "_"
             + std::to_string(omp_get_thread_num());
    std::ifstream rng(rng_name);
    if(!rng) { LBANN_ERROR("Failed to open ", rng_name); }
    rng >> ::generator;

    rng_name = dirname + "/rng_fast_generator_" + rank_in_trainer + "_"
             + std::to_string(omp_get_thread_num());
    std::ifstream rng_fast(rng_name);
    if(!rng_fast) { LBANN_ERROR("Failed to open ", rng_name); }
    rng_fast >> ::fast_generator;
   }
#else
    rng_name = dirname + "/rng_generator_" + rank_in_trainer;
    std::ifstream rng(rng_name);
    if(!rng) { LBANN_ERROR("Failed to open ", rng_name); }
    rng >> ::generator;

    rng_name = dirname + "/rng_fast_generator_" + rank_in_trainer;
    std::ifstream rng_fast(rng_name);
    if(!rng_fast) { LBANN_ERROR("Failed to open ", rng_name); }
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
      get_generator().seed(hash_combine(seed, omp_get_thread_num()));
      get_fast_generator().seed(hash_combine(seed, omp_get_thread_num()));
    }
#else
    get_generator().seed(seed);
    get_fast_generator().seed(seed);
#endif

#ifdef LBANN_SET_EL_RNG
    // Set Elemental's RNG seed
    auto elemental_seed = hash_combine(seed, 104729); // 10000th prime
    elemental_seed = (comm == nullptr
                      ? hash_combine(elemental_seed, El::mpi::Rank(El::mpi::COMM_WORLD))
                      : hash_combine(elemental_seed, comm->get_rank_in_trainer()));
    El::Generator().seed(elemental_seed);
#endif

  } else {
    // Seed with a random value.
    std::random_device rd;
    unsigned rand_val = rd();
#ifdef _OPENMP
    #pragma omp parallel
    {
      get_generator().seed(hash_combine(rand_val, omp_get_thread_num()));
      get_fast_generator().seed(hash_combine(rand_val, omp_get_thread_num()));
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

template <typename TensorDataType>
void gaussian_fill(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n,
                   TensorDataType mean, TensorDataType stddev) {
#ifndef LBANN_DETERMINISTIC
  El::Gaussian(mat, m, n, mean, stddev);
#else
  gaussian_fill_procdet(mat, m, n, mean, stddev);
#endif  // LBANN_DETERMINISTIC
}

template <typename TensorDataType>
void bernoulli_fill(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n, double p) {
#ifndef LBANN_DETERMINISTIC
  El::Bernoulli(mat, m, n, p);
#else
  bernoulli_fill_procdet(mat, m, n, p);
#endif  // LBANN_DETERMINISTIC
}

template <typename TensorDataType>
void uniform_fill(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n,
                  TensorDataType center, TensorDataType radius) {
#ifndef LBANN_DETERMINISTIC
  El::Uniform(mat, m, n, center, radius);
#else
  uniform_fill_procdet(mat, m, n, center, radius);
#endif  // LBANN_DETERMINISTIC
}

template <typename TensorDataType>
void gaussian_fill_procdet(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n,
                           TensorDataType mean, TensorDataType stddev) {
#if defined(LBANN_HAS_GPU_FP16) && defined(LBANN_HAS_HALF)
  using RandDataType = typename std::conditional<
    El::Or<std::is_same<TensorDataType,cpu_fp16>,
           std::is_same<TensorDataType,fp16>>::value,
    float, TensorDataType>::type;
#elif defined(LBANN_HAS_GPU_FP16)
  using RandDataType = typename std::conditional<
    std::is_same<TensorDataType,fp16>::value,
    float, TensorDataType>::type;
#elif defined(LBANN_HAS_HALF)
  using RandDataType = typename std::conditional<
    std::is_same<TensorDataType,cpu_fp16>::value,
    float, TensorDataType>::type;
#else
  using RandDataType = TensorDataType;
#endif // LBANN_HAS_GPU_FP16

  CircMatDT<RandDataType, El::Device::CPU> vals(m, n, mat.Grid(), 0);
  if (vals.Participating()) {
    auto& local_vals = vals.Matrix();
    auto& gen = get_generator();
    std::normal_distribution<RandDataType> dist(mean, stddev);
    for (El::Int col = 0; col < local_vals.Width(); ++col) {
      for (El::Int row = 0; row < local_vals.Height(); ++row) {
        local_vals(row, col) = dist(gen);
      }
    }
  }
  El::Copy(vals, mat);
}

template <typename TensorDataType>
void bernoulli_fill_procdet(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n, double p) {
  CircMatDT<TensorDataType, El::Device::CPU> vals(m, n, mat.Grid(), 0);
  if (vals.Participating()) {
    auto& local_vals = vals.Matrix();
    auto& gen = get_generator();
    std::bernoulli_distribution dist(p);
    for (El::Int col = 0; col < local_vals.Width(); ++col) {
      for (El::Int row = 0; row < local_vals.Height(); ++row) {
        local_vals(row, col) = El::To<TensorDataType>(dist(gen));
      }
    }
  }
  El::Copy(vals, mat);
}

template <typename TensorDataType>
void uniform_fill_procdet(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n,
                          TensorDataType center, TensorDataType radius) {
#if defined(LBANN_HAS_GPU_FP16) && defined(LBANN_HAS_HALF)
  using RandDataType = typename std::conditional<
    El::Or<std::is_same<TensorDataType,cpu_fp16>,
           std::is_same<TensorDataType,fp16>>::value,
    float, TensorDataType>::type;
#elif defined(LBANN_HAS_GPU_FP16)
  using RandDataType = typename std::conditional<
    std::is_same<TensorDataType,fp16>::value,
    float, TensorDataType>::type;
#elif defined(LBANN_HAS_HALF)
  using RandDataType = typename std::conditional<
    std::is_same<TensorDataType,cpu_fp16>::value,
    float, TensorDataType>::type;
#else
  using RandDataType = TensorDataType;
#endif // LBANN_HAS_GPU_FP16

  CircMatDT<RandDataType, El::Device::CPU> vals(m, n, mat.Grid(), 0);
  if (vals.Participating()) {
    auto& local_vals = vals.Matrix();
    auto& gen = get_generator();
    std::uniform_real_distribution<RandDataType> dist(center - radius,
                                                      center + radius);
    for (El::Int col = 0; col < local_vals.Width(); ++col) {
      for (El::Int row = 0; row < local_vals.Height(); ++row) {
        local_vals(row, col) = dist(gen);
      }
    }
  }
  El::Copy(vals, mat);
}

#define PROTO(T)                                                                                                  \
  template void gaussian_fill<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, T mean, T stddev);         \
  template void bernoulli_fill<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, double p);                \
  template void uniform_fill<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, T center, T radius);        \
  template void gaussian_fill_procdet<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, T mean, T stddev); \
  template void bernoulli_fill_procdet<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, double p);        \
  template void uniform_fill_procdet<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, T center, T radius)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace lbann
