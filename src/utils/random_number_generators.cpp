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
#include <lbann/utils/memory.hpp>
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

// Local index for the I/O generators
thread_local size_t local_io_generators_index = 0;
std::vector<lbann::io_rng_t> io_generators;
bool io_generators_inited = false;
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

int get_num_io_generators() {
  return ::io_generators.size();
}

locked_io_rng_ref set_io_generators_local_index(size_t idx) {
  ::local_io_generators_index = idx;
  if (!::io_generators_inited) { LBANN_ERROR("I/O RNG seed not set"); }
  return locked_io_rng_ref(::io_generators[idx]);
}

rng_gen& get_io_generator() {
  const size_t idx = ::local_io_generators_index;
  io_rng_t& io_rng = ::io_generators[idx];
  if (io_rng.active_thread_id != std::this_thread::get_id()) {
    LBANN_ERROR("I/O RNG illegal thread access");
  }
  return io_rng.io_generator;
}

fast_rng_gen& get_fast_io_generator() {
  const size_t idx = ::local_io_generators_index;
  io_rng_t& io_rng = ::io_generators[idx];
  if (io_rng.active_thread_id != std::this_thread::get_id()) {
    LBANN_ERROR("I/O RNG illegal thread access");
  }
  return io_rng.fast_io_generator;
}

void init_random(int seed, int num_io_RNGs, lbann_comm *comm) {
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

  init_io_random(seed, num_io_RNGs);
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

void init_io_random(int seed, int num_io_RNGs) {
  int seed_base = seed;
  if (seed == -1) {
    // Seed with a random value.
    std::random_device rd;
    seed_base = rd();
  }

  ::io_generators_inited = true;
  ::io_generators.resize(num_io_RNGs);
  for(int i = 0; i < num_io_RNGs; i++) {
    auto& io_rng = ::io_generators[i];
    io_rng.io_generator.seed(hash_combine(seed_base, i));
    io_rng.fast_io_generator.seed(hash_combine(seed_base, i));
    io_rng.io_mutex = make_unique<std::mutex>();
    io_rng.active_thread_id = std::thread::id();
  }
}

}  // namespace lbann
