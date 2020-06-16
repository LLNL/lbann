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
#include "lbann/utils/random_number_generators.hpp"
#include "lbann/utils/hash.hpp"
#include "lbann/utils/exception.hpp"
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

bool generator_inited = false;
bool fast_generator_inited = false;

thread_local lbann::rng_gen data_seq_generator;
thread_local bool data_seq_generator_inited = false;
int data_seq_generator_seed_base = 0;
bool data_seq_generator_seed_inited = false;

thread_local lbann::rng_gen io_generator;
thread_local bool io_generator_inited = false;
int io_generator_seed_base = 0;
bool io_generator_seed_inited = false;

thread_local lbann::fast_rng_gen fast_io_generator;
thread_local bool fast_io_generator_inited = false;
int fast_io_generator_seed_base = 0;
bool fast_io_generator_seed_inited = false;
}

namespace lbann {

rng_gen& get_generator() {
  if (!::generator_inited) { LBANN_ERROR("RNG seed not set"); }
  return ::generator;
}

fast_rng_gen& get_fast_generator() {
  if (!::fast_generator_inited) { LBANN_ERROR("Fast RNG seed not set"); }
  return ::fast_generator;
}

rng_gen& get_data_seq_generator() {
  if (!::data_seq_generator_inited) {
    if (!::data_seq_generator_seed_inited) { LBANN_ERROR("data sequence RNG seed not set"); }
    ::data_seq_generator.seed(::data_seq_generator_seed_base);
    ::data_seq_generator_inited = true;
  }
  return ::data_seq_generator;
}

rng_gen& get_io_generator() {
  if (!::io_generator_inited) {
    if (!::io_generator_seed_inited) { LBANN_ERROR("I/O RNG seed not set"); }
    ::io_generator.seed(hash_combine(::io_generator_seed_base,
                                     std::this_thread::get_id()));
    ::io_generator_inited = true;
  }
  return ::io_generator;
}

fast_rng_gen& get_fast_io_generator() {
  if (!::fast_io_generator_inited) {
    if (!::fast_io_generator_seed_inited) { LBANN_ERROR("Fast I/O RNG seed not set"); }
    ::fast_io_generator.seed(hash_combine(::fast_io_generator_seed_base,
                                          std::this_thread::get_id()));
    ::fast_io_generator_inited = true;
  }
  return ::fast_io_generator;
}

void init_io_generator(const int local_thread_id) {
  ::io_generator.seed(hash_combine(::io_generator_seed_base,
                                   local_thread_id));
  ::io_generator_inited = true;
}

void init_fast_io_generator(const int local_thread_id) {
  ::fast_io_generator.seed(hash_combine(::fast_io_generator_seed_base,
                                        local_thread_id));
  ::fast_io_generator_inited = true;
}

void init_random(int seed, lbann_comm *comm) {
  generator_inited = true;
  fast_generator_inited = true;
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
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if(mpi_initialized) {
      // If MPI is initialized mix in the rank to ensure that Hydrogen
      // has good RNGs.  Note that under some configurations LBANN
      // will not do this, so it is good to ensure that Hydrogen is
      // well seeded.
      elemental_seed = (comm == nullptr
                        ? hash_combine(elemental_seed, El::mpi::Rank(El::mpi::COMM_WORLD))
                        : hash_combine(elemental_seed, comm->get_rank_in_trainer()));
    }
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
  ::data_seq_generator_seed_inited = true;
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
  ::io_generator_seed_inited = true;
  /// Reset the init flag so that generator will reinitialize
  ::io_generator_inited = false;

  ::fast_io_generator_seed_base = seed;
  ::fast_io_generator_seed_inited = true;
  /// Reset the init flag so that generator will reinitialize
  ::fast_io_generator_inited = false;
}

}  // namespace lbann
