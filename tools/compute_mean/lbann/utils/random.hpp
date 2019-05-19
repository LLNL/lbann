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

#include "lbann/base.hpp"
#include <random>

namespace lbann {

typedef std::mt19937 rng_gen;  // Mersenne Twister
typedef std::minstd_rand fast_rng_gen;  // Minimum standard, LC

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
 * for shuffling the data samples within each mini-bathc
 * @note If compiling with OpenMP, this is stored in a threadprivate variable.
 */
rng_gen& get_data_seq_generator();

/**
 * Return random integers uniformly distributed in [0, max).
 * @param g C++ uniform random bit generator.
 * @param max Upper bound on the distribution.
 * @note It turns out that the GCC std::uniform_int_distribution is really
 * slow. That implementation is used by most compilers. This implementation
 * is roughly five times faster than that one.
 */
template <typename Generator, typename T>
inline T fast_rand_int(Generator& g, T max) {
  typename Generator::result_type x;
  do {
    x = g();
  } while (x >= (Generator::max() - Generator::max() % max));
  return x % max;
}

/**
 * Faster variant of fast_rand_int in the case that max is a power of 2.
 * Do not call this if max is not a power of 2.
 */
template <typename Generator, typename T>
inline T fast_rand_int_pow2(Generator& g, T max) {
  typename Generator::result_type x;
  max -= 1;
  const typename Generator::result_type upper = Generator::max() -
      (Generator::max() & (typename Generator::result_type) max);
  do {
    x = g();
  } while (x >= upper);
  return x & ((typename Generator::result_type) max);
}

/**
 * Initialize the random number generator (with optional seed).
 * @param comm If present, mixes the process's rank within the model into the
 * seed; if not, uses the MPI world rank.
 * @todo Support saving/restoring the generator's state. This is directly
 * supported via the >> and << operators on the generator (reading/writing
 * from/to a stream).
 */
void init_random(int seed = -1);

/**
 * Initialize a random number generator (with optional seed) that is
 * specifically used for sequencing the training / testing data
 * samples.  Using a separate RNG for the data sequences helps provide
 * a stable training result that does not vary with how much I/O
 * parallelism is applied.
 * @todo Support saving/restoring the generator's state. This is directly
 * supported via the >> and << operators on the generator (reading/writing
 * from/to a stream).
 */
void init_data_seq_random(int seed = -1);

template<typename DistType,typename DType=DataType>
class rng {

 private:
  DistType m_dist; // Distribution type

 public:
  typename DistType::result_type gen() {
    return m_dist(get_generator());
  }
  rng() { }
  // bernoulli_distribution with prob p
  rng(DType p) : m_dist(p) {}
  // (uniform) real distribution between min/mean and max/stdev
  rng(DType a,DType b) : m_dist(a,b) {}
};

}// end namespace
#endif // LBANN_UTILS_RNG_HPP
