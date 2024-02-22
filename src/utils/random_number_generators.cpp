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

#include "lbann/utils/random_number_generators.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/hash.hpp"
#include <lbann/utils/memory.hpp>
#include <omp.h>
#include <thread>

namespace {
#if not defined(LBANN_DETERMINISTIC) && defined(_OPENMP)
#ifdef __ICC
lbann::rng_gen OMP_generator;
#pragma omp threadprivate(OMP_generator)

lbann::fast_rng_gen OMP_fast_generator;
#pragma omp threadprivate(OMP_fast_generator)

bool OMP_generator_inited = false;
#pragma omp threadprivate(OMP_generator_inited)

bool OMP_fast_generator_inited = false;
#pragma omp threadprivate(OMP_fast_generator_inited)
#else // ! __ICC
// Random number generator, file-visible only.
// Defined like this to work around a GCC problem with threadprivate objects:
// https://stackoverflow.com/questions/23552077/how-to-define-a-object-or-struct-as-threadprivate-in-openmp/
extern lbann::rng_gen OMP_generator;
#pragma omp threadprivate(OMP_generator)
lbann::rng_gen OMP_generator;

extern lbann::fast_rng_gen OMP_fast_generator;
#pragma omp threadprivate(OMP_fast_generator)
lbann::fast_rng_gen OMP_fast_generator;

extern bool OMP_generator_inited;
#pragma omp threadprivate(OMP_generator_inited)
bool OMP_generator_inited = false;

extern bool OMP_fast_generator_inited;
#pragma omp threadprivate(OMP_fast_generator_inited)
bool OMP_fast_generator_inited = false;
#endif
#else // not defined(LBANN_DETERMINISTIC) && defined(_OPENMP)
lbann::rng_gen OMP_generator;
lbann::fast_rng_gen OMP_fast_generator;
bool OMP_generator_inited = false;
bool OMP_fast_generator_inited = false;
#endif

lbann::rng_gen generator;
lbann::fast_rng_gen fast_generator;
lbann::fast_rng_gen ltfb_generator;

bool generator_inited = false;
bool fast_generator_inited = false;
bool ltfb_generator_inited = false;

thread_local lbann::rng_gen data_seq_generator;
thread_local bool data_seq_generator_inited = false;
int data_seq_generator_seed_base = 0;
bool data_seq_generator_seed_inited = false;

// Local index for the I/O generators
thread_local size_t local_io_generators_index = 0;
std::vector<lbann::io_rng_t> io_generators;
bool io_generators_inited = false;
} // namespace

namespace lbann {

rng_gen& get_generator()
{
  if (!::generator_inited) {
    LBANN_ERROR("RNG seed not set");
  }
  return ::generator;
}

fast_rng_gen& get_fast_generator()
{
  if (!::fast_generator_inited) {
    LBANN_ERROR("Fast RNG seed not set");
  }
  return ::fast_generator;
}

fast_rng_gen& get_ltfb_generator()
{
  if (!::ltfb_generator_inited) {
    LBANN_ERROR("LTFB RNG seed not set");
  }
  return ::ltfb_generator;
}

rng_gen& get_OMP_generator()
{
  if (!::OMP_generator_inited) {
    LBANN_ERROR("OMP RNG seed not set");
  }
  return ::OMP_generator;
}

fast_rng_gen& get_OMP_fast_generator()
{
  if (!::OMP_fast_generator_inited) {
    LBANN_ERROR("OMP Fast RNG seed not set");
  }
  return ::OMP_fast_generator;
}

rng_gen& get_data_seq_generator()
{
  if (!::data_seq_generator_inited) {
    if (!::data_seq_generator_seed_inited) {
      LBANN_ERROR("data sequence RNG seed not set");
    }
    ::data_seq_generator.seed(::data_seq_generator_seed_base);
    ::data_seq_generator_inited = true;
  }
  return ::data_seq_generator;
}

int get_num_io_generators() { return ::io_generators.size(); }

locked_io_rng_ref set_io_generators_local_index(size_t idx, execution_mode mode)
{
  auto num_io_rngs = get_num_io_generators();
  if (mode == execution_mode::invalid || num_io_rngs <= 2) {
    // If there isn't a valid execution mode or there are too few RNGs
    // just use the whole range.
    ::local_io_generators_index = idx % num_io_rngs;
  }
  else {
    // If there is a valid execution mode provided
    // map training ranks to the lower half of the generator range and
    // all other execution modes use the upper half of the bank.
    // Keep this as one vector to make it easier to serialize and split
    // it between training and everything else to improve determinism
    int bank_split = num_io_rngs / 2;
    if (mode == execution_mode::training) {
      ::local_io_generators_index = idx % bank_split;
    }
    else {
      ::local_io_generators_index = bank_split + (idx % bank_split);
    }
  }
  if (!::io_generators_inited) {
    LBANN_ERROR("I/O RNG seed not set");
  }
  return locked_io_rng_ref(::io_generators[::local_io_generators_index]);
}

rng_gen& get_io_generator()
{
  const size_t idx = ::local_io_generators_index;
  io_rng_t& io_rng = ::io_generators[idx];
  if (io_rng.active_thread_id.load() != std::this_thread::get_id()) {
    LBANN_ERROR("I/O RNG illegal thread access");
  }
  return io_rng.generator;
}

fast_rng_gen& get_fast_io_generator()
{
  const size_t idx = ::local_io_generators_index;
  io_rng_t& io_rng = ::io_generators[idx];
  if (io_rng.active_thread_id.load() != std::this_thread::get_id()) {
    LBANN_ERROR("I/O RNG illegal thread access");
  }
  return io_rng.fast_generator;
}

void init_random(int seed, int num_io_RNGs, lbann_comm* comm)
{
  generator_inited = true;
  fast_generator_inited = true;
  OMP_generator_inited = true;
  OMP_fast_generator_inited = true;

  // Use different seed on each rank in trainer
  if (seed == -1) {
    std::random_device rd;
    seed = rd();
  }
  else if (comm != nullptr) {
    seed = hash_combine(seed, comm->get_rank_in_trainer());
  }
  else if (El::mpi::Initialized()) {
    seed = hash_combine(seed, El::mpi::Rank(El::mpi::COMM_WORLD));
  }

  get_generator().seed(seed);
  get_fast_generator().seed(hash_combine(seed, 41263)); // 4321th prime

  // Seed every OpenMP thread, if present.
  // Note: Threadprivate OMP variables don't work with dynamic threads.
#if not defined(LBANN_DETERMINISTIC) && defined(_OPENMP)
#pragma omp parallel
  {
    const int thread = omp_get_thread_num();
    const int thread_seed = hash_combine(seed, thread);
    OMP_generator_inited = true;
    OMP_fast_generator_inited = true;
    get_OMP_generator().seed(thread_seed);
    get_OMP_fast_generator().seed(
      hash_combine(thread_seed, 132241)); // 12345th prime
  }
#else // not defined(LBANN_DETERMINISTIC) && defined(_OPENMP)
  get_OMP_generator().seed(seed);
  get_OMP_fast_generator().seed(hash_combine(seed, 41263)); // 4321th prime
#endif

  // Set Elemental's RNG seed
  El::Generator().seed(hash_combine(seed, 104729)); // 10000th prime

  // Initialize IO RNGs
  init_io_random(seed, num_io_RNGs);
}

void init_data_seq_random(int seed)
{
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

void init_ltfb_random(int seed)
{
  if (seed == -1) {
    seed = 20201003;
  }

  ltfb_generator_inited = true;
  get_ltfb_generator().seed(seed);
}

void init_io_random(int seed, int num_io_RNGs)
{
  int seed_base = seed;
  if (seed == -1) {
    // Seed with a random value.
    std::random_device rd;
    seed_base = rd();
  }

  ::io_generators.resize(num_io_RNGs);
  for (int i = 0; i < num_io_RNGs; i++) {
    auto& io_rng = ::io_generators[i];
    io_rng.generator.seed(hash_combine(seed_base, i));
    io_rng.fast_generator.seed(hash_combine(seed_base, i));
    io_rng.active_thread_id.store(std::thread::id());
  }
  ::io_generators_inited = true;
}

} // namespace lbann
