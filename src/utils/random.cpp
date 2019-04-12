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

#include "lbann/utils/random.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/io/file_io.hpp"

namespace {
thread_local lbann::rng_gen generator;
thread_local bool generator_inited = false;
thread_local lbann::fast_rng_gen fast_generator;
thread_local bool fast_generator_inited = false;
thread_local lbann::rng_gen data_seq_generator;
thread_local bool data_seq_generator_inited = false;
int generator_seed_base = 0;
int fast_generator_seed_base = 0;
int data_seq_generator_seed_base = 0;
}

namespace lbann {

rng_gen& get_generator() {
  if (!::generator_inited) {
    std::hash<std::thread::id> h;
    ::generator.seed((::generator_seed_base << 8) |
                     h(std::this_thread::get_id()));
    ::generator_inited = true;
  }
  return ::generator;
}

fast_rng_gen& get_fast_generator() {
  if (!::fast_generator_inited) {
    std::hash<std::thread::id> h;
    ::fast_generator.seed((::fast_generator_seed_base << 8) |
                          h(std::this_thread::get_id()));
    ::fast_generator_inited = true;
  }
  return ::fast_generator;
}

rng_gen& get_data_seq_generator() {
  if (!::data_seq_generator_inited) {
    std::hash<std::thread::id> h;
    ::data_seq_generator.seed((::data_seq_generator_seed_base << 8) |
                              h(std::this_thread::get_id()));
    ::data_seq_generator_inited = true;
  }
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
  LBANN_WARNING("Checkpointing RNG state not supported.");
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

  LBANN_WARNING("Restoring RNG state not supported.");
  return true;
}

void init_random(int seed, lbann_comm *comm) {
  if (seed == -1) {
    // Seed with a random value.
    std::random_device rd;
    seed = rd();
  }
  ::generator_seed_base = seed;
  ::fast_generator_seed_base = seed;

#ifdef LBANN_SET_EL_RNG
  if (comm != nullptr) {
    El::Generator().seed(seed ^ comm->get_rank_in_trainer());
  } else {
    El::Generator().seed(seed ^ El::mpi::Rank(El::mpi::COMM_WORLD));
  }
#endif
}

void init_data_seq_random(int seed) {
  if (seed == -1) {
    // Seed with a random value.
    std::random_device rd;
    seed = rd();
  }
  ::data_seq_generator_seed_base = seed;
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
