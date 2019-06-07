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
#include "lbann/comm.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/exception.hpp"
#include <random>

namespace lbann {

/** Probability distributions. */
enum class probability_distribution {invalid, gaussian, bernoulli, uniform};

using rng_gen = std::mt19937;  // Mersenne Twister
using fast_rng_gen = std::minstd_rand;  // Minimum standard, LC

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
#ifdef LBANN_DEBUG
  if (max == 0) {
    LBANN_ERROR("fast_rand_int called with max=0");
  }
#endif
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

// Methods for quickly generating uniformly random values in [0, 1).

namespace details {

// See section on converting uint64_ts to doubles in:
// http://xoshiro.di.unimi.it/

template <typename Generator>
inline float random_float(Generator& g) {
  const uint32_t r = uint32_t(g()) >> 9;  // Truncate if needed.
  return r * (1.0f / 8388608.0f);
}

template <typename Generator>
inline double random_double_32(Generator& g) {
  const uint32_t r1 = g() >> 5;
  const uint32_t r2 = g() >> 6;
  return (r1 * 67108864.0 + r2) * (1.0 / 9007199254740992.0);
}

template <typename Generator>
inline double random_double_64(Generator& g) {
  const uint64_t r = g() >> 11;
  return r * (1.0 / 9007199254740992.0);
}

template <typename Generator>
inline double random_double(Generator& g) {
  // TODO: Replace with if constexpr when possible.
  if (sizeof(typename Generator::result_type) == 4) {
    return random_double_32(g);
  } else if (sizeof(typename Generator::result_type) == 8) {
    return random_double_64(g);
  } else {
    LBANN_ERROR("Unsupported generator type");
  }
}

template <typename Generator, typename T>
struct random_uniform_impl {
  static T generate(Generator&);
};
template <typename Generator>
struct random_uniform_impl<Generator, float> {
  static float generate(Generator& g) { return random_float(g); }
};
template <typename Generator>
struct random_uniform_impl<Generator, double> {
  static float generate(Generator& g) { return random_double(g); }
};

}  // namespace details

/** Generate uniformly random values in the range [0, 1). */
template <typename T, typename Generator>
inline T fast_random_uniform(Generator& g) {
  static_assert(sizeof(typename Generator::result_type) == 4 ||
                sizeof(typename Generator::result_type) == 8,
                "Invalid generator result_type.");
  return details::random_uniform_impl<Generator, T>::generate(g);
}

/** @brief Initialize the random number generator (with optional seed).
 *
 *  @param seed Seed value for the random number generator
 *  @param comm If present, mixes the process's rank within the model
 *              into the seed; if not, uses the MPI world rank.
 *
 */
void init_random(int seed = -1, lbann_comm *comm = nullptr);

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
 *
 * Called from init_random
 */
void init_io_random(int seed = -1);

/**
 * Make mat into an m x n matrix where each entry is independently drawn from
 * a Gaussian distribution with given mean and standard deviation.
 * Unless selected so at compile-time, this ensures the entries of the matrix do
 * not change as the grid it is distributed over changes; that is, it will have
 * the same entries when mat spans any number of processes.
 */
void gaussian_fill(AbsDistMat& mat, El::Int m, El::Int n, DataType mean = 0.0f,
                   DataType stddev = 1.0f);
/**
 * Make mat into an m x n matrix where each entry is an indepenent Bernoulli
 * random variable with parameter p.
 * This makes the same guarantees as gaussian_fill.
 */
void bernoulli_fill(AbsDistMat& mat, El::Int m, El::Int n, double p = 0.5);
/**
 * Make mat into an m x n matrix where each entry is independently uniformly
 * sampled from a ball with the given center and radius.
 * This makes the same guarantees as gaussian_fill.
 */
void uniform_fill(AbsDistMat& mat, El::Int m, El::Int n, DataType center = 0.0f,
                  DataType radius = 1.0f);

/**
 * Make mat into an m x n matrix where each entry is independently drawn from
 * a Gaussian distribution with given mean and standard deviation.
 * This always ensures that the entries of the matrix do not change as the grid
 * it is distributed over changes.
 */
void gaussian_fill_procdet(AbsDistMat& mat, El::Int m, El::Int n,
                           DataType mean = 0.0f, DataType stddev = 1.0f);
/**
 * Make mat into an m x n matrix where each entry is an independent Bernoulli
 * random variable with parameter p.
 * This makes the same guarantees as gaussian_fill_procdet.
 */
void bernoulli_fill_procdet(AbsDistMat& mat, El::Int m, El::Int n, double p = 0.5);
/**
 * Make mat into an m x n matrix where each entry is independently uniformly
 * sampled from a ball with the given center and radius.
 * This makes the same guarantees as gaussian_fill_procdet.
 */
void uniform_fill_procdet(AbsDistMat& mat, El::Int m, El::Int n,
                          DataType center = 0.0f, DataType radius = 1.0f);

bool save_rng_to_checkpoint_shared(persist& p, const lbann_comm* comm);
bool load_rng_from_checkpoint_shared(persist& p, const lbann_comm* comm);

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

/** Multiply entries of distributed matrix  with
 * a multiplier generated according to bernoulli_distribution
 */
template <typename DType=DataType>
void rng_bernoulli(const float p, DistMat *m) {

  /// the scale for undropped inputs at training time given as @f$ 1 / (1 - p) @f$
  float scale = 1. / (1. - p);

  //@todo: use seed from parameter
  rng<std::bernoulli_distribution,DType> myrn(p); //magic seed?

  for (int row = 0; row < m->LocalHeight(); ++row) {
    for (int col = 0; col < m->LocalWidth(); ++col) {
      m->Set(row,col,myrn.gen()*scale); //SetLocal?
    }
  }
}


}// end namespace
#endif // LBANN_UTILS_RNG_HPP
