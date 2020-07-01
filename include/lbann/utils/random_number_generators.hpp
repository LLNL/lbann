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

#ifndef LBANN_UTILS_RNG_HPP
#define LBANN_UTILS_RNG_HPP

#include "lbann/comm.hpp"
#include <random>
#include <mutex>
#include <thread>

namespace lbann {

using rng_gen = std::mt19937;  // Mersenne Twister
using fast_rng_gen = std::minstd_rand;  // Minimum standard, LC

struct io_rng_t {
  lbann::rng_gen io_generator;
  lbann::fast_rng_gen fast_io_generator;
  std::unique_ptr<std::mutex> io_mutex;
  // Track the owner so that it is easy to ensure the right thread is
  // using this structure.
  std::thread::id active_thread_id;
};

/**
 * Return a reference to the global LBANN random number generator.
 * @note If compiling with OpenMP, this is stored in a threadprivate variable.
 */
rng_gen& get_generator();

/**
 * Return a reference to a possibly-faster global LBANN random number generator.
 * Compared to get_generator, this should be slightly faster.
 * @note If compiling with OpenMP, this is stored in a threadprivate variable.
 */
fast_rng_gen& get_fast_generator();

/**
 * Return a reference to the global LBANN random number generator used
 * for shuffling the data samples within each mini-batch
 * @note This is stored in a thread_local variable.
 */
rng_gen& get_data_seq_generator();

/** @brief Returns the number of provisioned I/O generators. */
int get_num_io_generators();

/** @brief Sets the local index for a thread to access the correct I/O RNGs. */
io_rng_t& set_io_generators_local_index(size_t idx);

/**
 * Return a reference to the global LBANN random number generator used
 * for shuffling the data samples within each mini-batch
 * @note This is stored in a thread_local variable.
 */
rng_gen& get_io_generator();

/**
 * Return a reference to the fast global LBANN random number generator used
 * for the I/O threads
 * @note This is stored in a thread_local variable.
 */
fast_rng_gen& get_fast_io_generator();

/** @brief Initialize the random number generator (with optional seed).
 *
 *  @param seed Seed value for the random number generator
 *  @param comm If present, mixes the process's rank within the model
 *              into the seed; if not, uses the MPI world rank.
 *
 */
void init_random(int seed = -1, int num_io_RNGs = 1, lbann_comm *comm = nullptr);

/**
 * Initialize a random number generator (with optional seed) that is
 * specifically used for sequencing the training / testing data
 * samples.  Using a separate RNG for the data sequences helps provide
 * a stable training result that does not vary with how much I/O
 * parallelism is applied.
 */
void init_data_seq_random(int seed = -1);

/**
 * Initialize a random number generator (with optional seed) that is
 * specifically used by the I/O threads for tasks such as data
 * preprocessing, etc.
 * Includes the number of I/O RNGs required.
 *
 * Called from init_random
 */
void init_io_random(int seed = -1, int num_io_RNGs = 1);

} // namespace lbann

#endif // LBANN_UTILS_RNG_HPP
