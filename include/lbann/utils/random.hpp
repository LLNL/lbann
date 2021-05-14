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

#ifndef LBANN_UTILS_RANDOM_HPP
#define LBANN_UTILS_RANDOM_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/random_number_generators.hpp"

namespace lbann {

/** Probability distributions. */
enum class probability_distribution {invalid, gaussian, bernoulli, uniform};

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

// Methods for generating uniformly random values in [0, 1).

namespace details {

/** Generates uniform random value in the range [0, 1). The generator
 *  is assumed to produce at least 32 random bits.
 */
template <typename Generator, typename T>
struct random_uniform_impl {
  static T generate(Generator&);
};
template <typename Generator>
struct random_uniform_impl<Generator, float> {
  static float generate(Generator& g) {
    // float has a 24-bit significand, including an implicit bit. See
    // section on converting uint64_ts to doubles in
    // http://xoshiro.di.unimi.it/
    constexpr uint64_t mask32 = 0xFFFFFFFFull;
    const uint64_t r = uint64_t(g()) & mask32;
    return (r >> 8) * (1.0f / 16777216.0f);
  }
};
template <typename Generator>
struct random_uniform_impl<Generator, double> {
  static double generate(Generator& g) {
    // double has a 53-bit significand, including an implicit bit. See
    // section on converting uint64_ts to doubles in
    // http://xoshiro.di.unimi.it/
    constexpr uint64_t mask32 = 0xFFFFFFFFull;
    const uint64_t r = (uint64_t(g()) << 32) | (uint64_t(g()) & mask32);
    return (r >> 11) * (1.0 / 9007199254740992.0);
  }
};

} // namespace details

/** @brief Generate uniform random value in the range [0, 1). */
template <typename T, typename Generator>
inline T random_uniform(Generator& g) {
  return details::random_uniform_impl<Generator, T>::generate(g);
}

/**
 * Make mat into an m x n matrix where each entry is independently drawn from
 * a Gaussian distribution with given mean and standard deviation.
 * Unless selected so at compile-time, this ensures the entries of the matrix do
 * not change as the grid it is distributed over changes; that is, it will have
 * the same entries when mat spans any number of processes.
 */
template <typename TensorDataType>
void gaussian_fill(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n, TensorDataType mean = 0.0,
                   TensorDataType stddev = 1.0);
/**
 * Make mat into an m x n matrix where each entry is an indepenent Bernoulli
 * random variable with parameter p.
 * This makes the same guarantees as gaussian_fill.
 */
template <typename TensorDataType>
void bernoulli_fill(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n, double p = 0.5);
/**
 * Make mat into an m x n matrix where each entry is independently uniformly
 * sampled from a ball with the given center and radius.
 * This makes the same guarantees as gaussian_fill.
 */
template <typename TensorDataType>
void uniform_fill(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n, TensorDataType center = 0.0,
                  TensorDataType radius = 1.0);

/**
 * Make mat into an m x n matrix where each entry is independently drawn from
 * a Gaussian distribution with given mean and standard deviation.
 * This always ensures that the entries of the matrix do not change as the grid
 * it is distributed over changes.
 */
template <typename TensorDataType>
void gaussian_fill_procdet(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n,
                           TensorDataType mean = 0.0, TensorDataType stddev = 1.0);
/**
 * Make mat into an m x n matrix where each entry is an independent Bernoulli
 * random variable with parameter p.
 * This makes the same guarantees as gaussian_fill_procdet.
 */
template <typename TensorDataType>
void bernoulli_fill_procdet(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n, double p = 0.5);
/**
 * Make mat into an m x n matrix where each entry is independently uniformly
 * sampled from a ball with the given center and radius.
 * This makes the same guarantees as gaussian_fill_procdet.
 */
template <typename TensorDataType>
void uniform_fill_procdet(El::AbstractDistMatrix<TensorDataType>& mat, El::Int m, El::Int n,
                          TensorDataType center = 0.0, TensorDataType radius = 1.0);

/**
 * Make mat into an m x n matrix where each entry is independently
 * drawn from a Gaussian distribution with given mean and standard
 * deviation. Entries are generated in parallel, so there are no
 * guarantees of thread/process indendence.
 */
template <typename TensorDataType>
void gaussian_fill_parallel(
  El::AbstractDistMatrix<TensorDataType>& mat,
  El::Int m,
  El::Int n,
  TensorDataType mean = 0.0,
  TensorDataType stddev = 1.0);

bool save_rng_to_checkpoint_shared(persist& p, lbann_comm* comm);
bool save_rng_to_checkpoint_distributed(persist& p, lbann_comm* comm);
bool load_rng_from_checkpoint(persist& p, const lbann_comm* comm);

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

#ifndef LBANN_RANDOM_INSTANTIATE
#define PROTO(T)                                                                                                         \
  extern template void gaussian_fill<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, T mean, T stddev);         \
  extern template void bernoulli_fill<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, double p);                \
  extern template void uniform_fill<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, T center, T radius);        \
  extern template void gaussian_fill_procdet<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, T mean, T stddev); \
  extern template void bernoulli_fill_procdet<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, double p);        \
  extern template void uniform_fill_procdet<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, T center, T radius); \
  extern template void gaussian_fill_parallel<T>(El::AbstractDistMatrix<T>& mat, El::Int m, El::Int n, T mean, T stddev)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_RANDOM_INSTANTIATE

}// end namespace
#endif // LBANN_UTILS_RANDOM_HPP
