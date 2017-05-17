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
//
// patchworks_rand.hpp - LBANN PATCHWORKS header for random number generator
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS header for random number generator 
 *  - includes mt19937_64 based random number generator wrapper
 *  - used to create the random patch position jitter
 *  - require c++11
 */

#ifndef _PATCHWORKS_RAND_H_INCLUDED_
#define _PATCHWORKS_RAND_H_INCLUDED_
#include <chrono>
#include <random>
#include <map>

namespace lbann {
namespace patchworks {

/**
 * This wrapper contains one mt19937_64 generator, and use it to generate
 * random numbers from three different distributions including
 * - normal distribution of real numbers
 * - uniform distribution of integer numbers
 * - uniform distribution of real numbers
 * The real number type and the integer type must be given as the template parameters.
 * If not, by default, double and int types are used respectively.
 */
template <typename RealT = double, typename IntT = int>
class rand_patch {
 public:
  typedef std::mt19937_64 generator_t;
  typedef std::normal_distribution<RealT> dist_normal_t;
  typedef std::uniform_int_distribution<IntT> dist_uniform_int_t;
  typedef std::uniform_real_distribution<RealT> dist_uniform_real_t;

 protected:
  generator_t generator; ///< random number generator
  /// normal distribution of real numbers
  std::map<int, dist_normal_t> dist_normal;
  /// uniform distribution of integer numbers
  std::map<int, dist_uniform_int_t> dist_uniform_int;
  /// uniform distribution of real numbers
  std::map<int, dist_uniform_real_t> dist_uniform_real;

 public:
  rand_patch(void) { reset(); } ///< construct with random seeding
  /// construct by initializing the RNG using the given seed
  rand_patch(const generator_t::result_type _seed) { reset(_seed); }

  /// rely on the current clock to seed randomly
  void reset(void);
  /// use the given value to see the random number generator
  void reset(const generator_t::result_type _seed);

  /// initialize the normal distribution of real numbers with mean 'avg' and standard deviation 'dev'
  void init_normal(const int i, const RealT avg, const RealT dev);
  /// pull one real number from the normal distribution
  RealT gen_normal(const int i);

  /// initialize the uniform distribution of integer numbers in the range [min max]
  void init_uniform_int(const int i, const IntT min, const IntT max);
  /// pull one integer number from the uniform distribution
  IntT gen_uniform_int(const int i);

  /// initialize the uniform distribution of real numbers in the range [min max]
  void init_uniform_real(const int i, const RealT min, const RealT max);
  /// pull one real number from the uniform distribution
  RealT gen_uniform_real(const int i);
};

template <typename RealT, typename IntT>
inline void rand_patch<RealT, IntT>::reset(void)
{
  typedef std::chrono::high_resolution_clock myclock;
  const myclock::time_point beginning = myclock::now();
  const myclock::duration d = myclock::now() - beginning;
  const generator_t::result_type _seed = static_cast<generator_t::result_type>(d.count());
  generator.seed(_seed);
}

template <typename RealT, typename IntT>
inline void rand_patch<RealT, IntT>::reset(const generator_t::result_type _seed)
{
  generator.seed(_seed);
}

template <typename RealT, typename IntT>
inline void rand_patch<RealT, IntT>::init_normal(const int i, const RealT avg, const RealT dev)
{
  dist_normal[i] = dist_normal_t(avg, dev);
}

template <typename RealT, typename IntT>
inline RealT rand_patch<RealT, IntT>::gen_normal(const int i)
{
  return dist_normal[i](generator);
}

template <typename RealT, typename IntT>
inline void rand_patch<RealT, IntT>::init_uniform_int(const int i, const IntT min, const IntT max)
{
  dist_uniform_int[i] = dist_uniform_int_t(min, max);
}

template <typename RealT, typename IntT>
inline IntT rand_patch<RealT, IntT>::gen_uniform_int(const int i)
{
  return dist_uniform_int[i](generator);
}

template <typename RealT, typename IntT>
inline void rand_patch<RealT, IntT>::init_uniform_real(const int i, const RealT min, const RealT max)
{
  dist_uniform_real[i] = dist_uniform_real_t(min, max);
}

template <typename RealT, typename IntT>
inline RealT rand_patch<RealT, IntT>::gen_uniform_real(const int i)
{
  return dist_uniform_real[i](generator);
}

} // end of namespace patchworks
} // end of namespace lbann
#endif // _PATCHWORKS_RAND_H_INCLUDED_
