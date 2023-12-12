////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#define LBANN_RANDOM_INSTANTIATE
#include "lbann/utils/random.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/utils/hash.hpp"
#include "lbann/utils/profiling.hpp"

#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

namespace lbann {

namespace {
void save_rng_state(std::string rng_name, rng_gen& gen)
{
  std::ofstream rng(rng_name);
  if (!rng) {
    LBANN_ERROR("Failed to open ", rng_name);
  }
  rng << gen;
  rng.close();
}
void save_rng_state(std::string rng_name, fast_rng_gen& gen)
{
  std::ofstream rng(rng_name);
  if (!rng) {
    LBANN_ERROR("Failed to open ", rng_name);
  }
  rng << gen;
  rng.close();
}

void load_rng_state(std::string rng_name, rng_gen& gen)
{
  std::ifstream rng_seq(rng_name);
  if (!rng_seq) {
    LBANN_ERROR("Failed to open ", rng_name);
  }
  rng_seq >> gen;
}

void load_rng_state(std::string rng_name, fast_rng_gen& gen)
{
  std::ifstream rng_seq(rng_name);
  if (!rng_seq) {
    LBANN_ERROR("Failed to open ", rng_name);
  }
  rng_seq >> gen;
}
} // namespace

bool save_rng_to_checkpoint(persist& p, lbann_comm* comm, bool is_distributed)
{
  std::string dirname = std::string(p.m_checkpoint_dir) + "/rng_state";
  std::string rank_in_trainer;
  std::string rng_name;

  if (comm == nullptr) {
    rank_in_trainer = std::to_string(El::mpi::Rank(El::mpi::COMM_WORLD));
    makedir(dirname.c_str());
  }
  else {
    rank_in_trainer = std::to_string(comm->get_rank_in_trainer());
    if (comm->am_trainer_master() || is_distributed) {
      makedir(dirname.c_str());
    }
    comm->trainer_barrier();
  }

  if (comm == nullptr || comm->am_trainer_master() || is_distributed) {
    /// @todo - Note that the RNG with thread local data is not correct
    rng_name = dirname + "/rng_seq_generator";
    save_rng_state(rng_name, get_data_seq_generator());

    rng_name = dirname + "/EL_generator";
    std::ofstream rng_EL(rng_name);
    if (!rng_EL) {
      LBANN_ERROR("Failed to open ", rng_name);
    }
    rng_EL << El::Generator();
    rng_EL.close();
  }

  for (int i = 0; i < get_num_io_generators(); i++) {
    rng_name = dirname + "/rng_io_generator_" + rank_in_trainer + "_t" +
               std::to_string(i);
    std::ofstream rng_io(rng_name);
    if (!rng_io) {
      LBANN_ERROR("Failed to open ", rng_name);
    }
    rng_name = dirname + "/rng_fast_io_generator_" + rank_in_trainer + "_t" +
               std::to_string(i);
    std::ofstream rng_fast_io(rng_name);
    if (!rng_fast_io) {
      LBANN_ERROR("Failed to open ", rng_name);
    }

    locked_io_rng_ref io_rng = set_io_generators_local_index(i);
    // save_rng_state(rng_name);
    // save_rng_state(rng_name);
    rng_io << get_io_generator();
    rng_fast_io << get_fast_io_generator();

    rng_io.close();
    rng_fast_io.close();
  }

  rng_name = dirname + "/rng_generator_" + rank_in_trainer;
  save_rng_state(rng_name, get_generator());

  rng_name = dirname + "/rng_fast_generator_" + rank_in_trainer;
  save_rng_state(rng_name, get_fast_generator());

  rng_name = dirname + "/rng_ltfb_generator_" + rank_in_trainer;
  save_rng_state(rng_name, get_ltfb_generator());

#if not defined(LBANN_DETERMINISTIC) && defined(_OPENMP)
  // #ifdef _OPENMP
#pragma omp parallel private(rng_name)
  {
    rng_name = dirname + "/rng_OMP_generator_" + rank_in_trainer + "_" +
               std::to_string(omp_get_thread_num());
    save_rng_state(rng_name, get_OMP_generator());

    rng_name = dirname + "/rng_OMP_fast_generator_" + rank_in_trainer + "_" +
               std::to_string(omp_get_thread_num());
    save_rng_state(rng_name, get_OMP_fast_generator());
  }
#else // not defined(LBANN_DETERMINISTIC) && defined(_OPENMP)
  rng_name = dirname + "/rng_OMP_generator_" + rank_in_trainer;
  save_rng_state(rng_name, get_OMP_generator());

  rng_name = dirname + "/rng_OMP_fast_generator_" + rank_in_trainer;
  save_rng_state(rng_name, get_OMP_fast_generator());
#endif

  return true;
}

bool save_rng_to_checkpoint_shared(persist& p, lbann_comm* comm)
{
  return save_rng_to_checkpoint(p, comm, false);
}

bool save_rng_to_checkpoint_distributed(persist& p, lbann_comm* comm)
{
  return save_rng_to_checkpoint(p, comm, true);
}

bool load_rng_from_checkpoint(persist& p, const lbann_comm* comm)
{

  std::string dirname = std::string(p.m_checkpoint_dir) + "/rng_state";
  std::string rng_name;

  /// @todo - Note that the RNG with thread local data is not correct
  rng_name = dirname + "/rng_seq_generator";
  load_rng_state(rng_name, get_data_seq_generator());

  rng_name = dirname + "/EL_generator";
  std::ifstream rng_EL(rng_name);
  if (!rng_EL) {
    LBANN_ERROR("Failed to open ", rng_name);
  }
  rng_EL >> El::Generator();

  std::string rank_in_trainer;
  if (comm == nullptr) {
    rank_in_trainer = std::to_string(El::mpi::Rank(El::mpi::COMM_WORLD));
  }
  else {
    rank_in_trainer = std::to_string(comm->get_rank_in_trainer());
  }

  for (int i = 0; i < get_num_io_generators(); i++) {
    rng_name = dirname + "/rng_io_generator_" + rank_in_trainer + "_t" +
               std::to_string(i);
    std::ifstream rng_io(rng_name);
    if (!rng_io) {
      LBANN_ERROR("Failed to open ", rng_name);
    }
    rng_name = dirname + "/rng_fast_io_generator_" + rank_in_trainer + "_t" +
               std::to_string(i);
    std::ifstream rng_fast_io(rng_name);
    if (!rng_fast_io) {
      LBANN_ERROR("Failed to open ", rng_name);
    }

    locked_io_rng_ref io_rng = set_io_generators_local_index(i);
    rng_io >> get_io_generator();
    rng_fast_io >> get_fast_io_generator();
  }

  rng_name = dirname + "/rng_generator_" + rank_in_trainer;
  load_rng_state(rng_name, get_generator());

  rng_name = dirname + "/rng_fast_generator_" + rank_in_trainer;
  load_rng_state(rng_name, get_fast_generator());

  rng_name = dirname + "/rng_ltfb_generator_" + rank_in_trainer;
  load_rng_state(rng_name, get_ltfb_generator());

#if not defined(LBANN_DETERMINISTIC) && defined(_OPENMP)
#pragma omp parallel private(rng_name)
  {
    rng_name = dirname + "/rng_OMP_generator_" + rank_in_trainer + "_" +
               std::to_string(omp_get_thread_num());
    load_rng_state(rng_name, get_OMP_generator());

    rng_name = dirname + "/rng_OMP_fast_generator_" + rank_in_trainer + "_" +
               std::to_string(omp_get_thread_num());
    load_rng_state(rng_name, get_OMP_fast_generator());
  }
#else // not defined(LBANN_DETERMINISTIC) && defined(_OPENMP)
  rng_name = dirname + "/rng_OMP_generator_" + rank_in_trainer;
  load_rng_state(rng_name, get_OMP_generator());

  rng_name = dirname + "/rng_OMP_fast_generator_" + rank_in_trainer;
  load_rng_state(rng_name, get_OMP_fast_generator());
#endif
  return true;
}

template <typename TensorDataType>
void gaussian_fill(El::AbstractDistMatrix<TensorDataType>& mat,
                   El::Int m,
                   El::Int n,
                   TensorDataType mean,
                   TensorDataType stddev)
{
#ifdef LBANN_DETERMINISTIC
  gaussian_fill_procdet(mat, m, n, mean, stddev);
#else
  gaussian_fill_parallel(mat, m, n, mean, stddev);
#endif // LBANN_DETERMINISTIC
}

template <typename TensorDataType>
void bernoulli_fill(El::AbstractDistMatrix<TensorDataType>& mat,
                    El::Int m,
                    El::Int n,
                    double p)
{
#ifndef LBANN_DETERMINISTIC
  El::Bernoulli(mat, m, n, p);
#else
  bernoulli_fill_procdet(mat, m, n, p);
#endif // LBANN_DETERMINISTIC
}

template <typename TensorDataType>
void uniform_fill(El::AbstractDistMatrix<TensorDataType>& mat,
                  El::Int m,
                  El::Int n,
                  TensorDataType center,
                  TensorDataType radius)
{
#ifndef LBANN_DETERMINISTIC
  El::Uniform(mat, m, n, center, radius);
#else
  uniform_fill_procdet(mat, m, n, center, radius);
#endif // LBANN_DETERMINISTIC
}

template <typename TensorDataType>
void gaussian_fill_procdet(El::AbstractDistMatrix<TensorDataType>& mat,
                           El::Int m,
                           El::Int n,
                           TensorDataType mean,
                           TensorDataType stddev)
{
  LBANN_CALIPER_MARK_FUNCTION;
#if defined(LBANN_HAS_GPU_FP16) && defined(LBANN_HAS_HALF)
  using RandDataType =
    typename std::conditional<El::Or<std::is_same<TensorDataType, cpu_fp16>,
                                     std::is_same<TensorDataType, fp16>>::value,
                              float,
                              TensorDataType>::type;
#elif defined(LBANN_HAS_GPU_FP16)
  using RandDataType =
    typename std::conditional<std::is_same<TensorDataType, fp16>::value,
                              float,
                              TensorDataType>::type;
#elif defined(LBANN_HAS_HALF)
  using RandDataType =
    typename std::conditional<std::is_same<TensorDataType, cpu_fp16>::value,
                              float,
                              TensorDataType>::type;
#else
  using RandDataType = TensorDataType;
#endif // LBANN_HAS_GPU_FP16

  using CircMatType = El::
    DistMatrix<RandDataType, El::CIRC, El::CIRC, El::ELEMENT, El::Device::CPU>;
  CircMatType vals(m, n, mat.Grid(), 0);
  if (vals.Participating()) {
    auto* __restrict__ buffer = vals.Buffer(); // Should be contiguous
    const size_t size = vals.LocalHeight() * vals.LocalWidth();
    std::normal_distribution<RandDataType> dist(mean, stddev);
    auto& gen = get_generator();
    for (size_t i = 0; i < size; ++i) {
      buffer[i] = dist(gen);
    }
  }
  El::Copy(vals, mat);
}

template <typename TensorDataType>
void bernoulli_fill_procdet(El::AbstractDistMatrix<TensorDataType>& mat,
                            El::Int m,
                            El::Int n,
                            double p)
{
  LBANN_CALIPER_MARK_FUNCTION;
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
void uniform_fill_procdet(El::AbstractDistMatrix<TensorDataType>& mat,
                          El::Int m,
                          El::Int n,
                          TensorDataType center,
                          TensorDataType radius)
{
  LBANN_CALIPER_MARK_FUNCTION;
#if defined(LBANN_HAS_GPU_FP16) && defined(LBANN_HAS_HALF)
  using RandDataType =
    typename std::conditional<El::Or<std::is_same<TensorDataType, cpu_fp16>,
                                     std::is_same<TensorDataType, fp16>>::value,
                              float,
                              TensorDataType>::type;
#elif defined(LBANN_HAS_GPU_FP16)
  using RandDataType =
    typename std::conditional<std::is_same<TensorDataType, fp16>::value,
                              float,
                              TensorDataType>::type;
#elif defined(LBANN_HAS_HALF)
  using RandDataType =
    typename std::conditional<std::is_same<TensorDataType, cpu_fp16>::value,
                              float,
                              TensorDataType>::type;
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

template <typename TensorDataType>
void gaussian_fill_parallel(El::AbstractDistMatrix<TensorDataType>& mat,
                            El::Int m,
                            El::Int n,
                            TensorDataType mean,
                            TensorDataType stddev)
{
  LBANN_CALIPER_MARK_FUNCTION;
  // Type for generating random variables
#if defined(LBANN_HAS_GPU_FP16) && defined(LBANN_HAS_HALF)
  using RandDataType =
    typename std::conditional<El::Or<std::is_same<TensorDataType, cpu_fp16>,
                                     std::is_same<TensorDataType, fp16>>::value,
                              float,
                              TensorDataType>::type;
#elif defined(LBANN_HAS_GPU_FP16)
  using RandDataType =
    typename std::conditional<std::is_same<TensorDataType, fp16>::value,
                              float,
                              TensorDataType>::type;
#elif defined(LBANN_HAS_HALF)
  using RandDataType =
    typename std::conditional<std::is_same<TensorDataType, cpu_fp16>::value,
                              float,
                              TensorDataType>::type;
#else
  using RandDataType = TensorDataType;
#endif // LBANN_HAS_GPU_FP16

  // Resize matrix
  mat.Resize(m, n);

  // Nothing to be done if there is no local data
  if (mat.LockedMatrix().IsEmpty()) {
    return;
  }

  // Generate random numbers on root rank of redundant comm
  if (mat.RedundantRank() == 0) {

    // Local buffer to hold random variables
    using LocalMatType = El::Matrix<RandDataType, El::Device::CPU>;
    LocalMatType local_vals;
    if constexpr (std::is_same<TensorDataType, RandDataType>::value) {
      if (mat.GetLocalDevice() == El::Device::CPU) {
        El::View(local_vals, mat.Matrix());
      }
    }

    if (!local_vals.Viewing()) {
      local_vals.Resize(mat.LocalHeight(), mat.LocalWidth());
    }

    // Populate local buffer with random variables
    // Note: Need to duplicate distribution on each thread since GCC
    // STL uses stateful Marsaglia polar method
    std::normal_distribution<RandDataType> dist(mean, stddev);
    if (local_vals.Contiguous()) {
      auto* __restrict__ buffer = local_vals.Buffer();
      const size_t size = local_vals.Height() * local_vals.Width();
      LBANN_OMP_PARALLEL_FOR_ARGS(firstprivate(dist))
      for (size_t i = 0; i < size; ++i) {
        buffer[i] = dist(get_OMP_generator());
      }
    }
    else {
      auto* __restrict__ buffer = local_vals.Buffer();
      const size_t height = local_vals.Height();
      const size_t width = local_vals.Width();
      const size_t ldim = local_vals.LDim();
      LBANN_OMP_PARALLEL_FOR_ARGS(collapse(2) firstprivate(dist))
      for (size_t j = 0; j < width; ++j) {
        for (size_t i = 0; i < height; ++i) {
          buffer[i + j * ldim] = dist(get_OMP_generator());
        }
      }
    }

    // Copy to output matrix if needed
    if (!local_vals.Viewing()) {
      El::Copy(local_vals, mat.Matrix());
    }
  }

  // Make sure local matrix is identical across redundant comm
  El::Broadcast(static_cast<El::AbstractMatrix<TensorDataType>&>(mat.Matrix()),
                mat.RedundantComm(),
                0);
}

#define PROTO(T)                                                               \
  template void gaussian_fill<T>(El::AbstractDistMatrix<T> & mat,              \
                                 El::Int m,                                    \
                                 El::Int n,                                    \
                                 T mean,                                       \
                                 T stddev);                                    \
  template void bernoulli_fill<T>(El::AbstractDistMatrix<T> & mat,             \
                                  El::Int m,                                   \
                                  El::Int n,                                   \
                                  double p);                                   \
  template void uniform_fill<T>(El::AbstractDistMatrix<T> & mat,               \
                                El::Int m,                                     \
                                El::Int n,                                     \
                                T center,                                      \
                                T radius);                                     \
  template void gaussian_fill_procdet<T>(El::AbstractDistMatrix<T> & mat,      \
                                         El::Int m,                            \
                                         El::Int n,                            \
                                         T mean,                               \
                                         T stddev);                            \
  template void bernoulli_fill_procdet<T>(El::AbstractDistMatrix<T> & mat,     \
                                          El::Int m,                           \
                                          El::Int n,                           \
                                          double p);                           \
  template void uniform_fill_procdet<T>(El::AbstractDistMatrix<T> & mat,       \
                                        El::Int m,                             \
                                        El::Int n,                             \
                                        T center,                              \
                                        T radius);                             \
  template void gaussian_fill_parallel<T>(El::AbstractDistMatrix<T> & mat,     \
                                          El::Int m,                           \
                                          El::Int n,                           \
                                          T mean,                              \
                                          T stddev)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
