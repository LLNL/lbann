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

#ifndef LBANN_UTILS_RNG_HPP
#define LBANN_UTILS_RNG_HPP

#include "lbann/lbann_base.hpp"
#include <random>

namespace lbann {

typedef std::mt19937 rng_gen;  // Mersenne Twister

/**
 * Return a reference to the global LBANN random number generator.
 * @note If it matters, the generator is not thread-safe.
 */
rng_gen& get_generator();

/**
 * Initialize the random number generator (with optional seed).
 * @todo Support saving/restoring the generator's state. This is directly
 * supported via the >> and << operators on the generator (reading/writing
 * from/to a stream).
 */
void init_random(int seed = -1);

template<typename DistType,typename DType=DataType>
class rng {

  private:
    DistType m_dist; // Distribution type

  public:
  typename DistType::result_type gen() { return m_dist(get_generator()); }
    rng(){ }
    // bernoulli_distribution with prob p
  rng(DType p) : m_dist(p) {}
    // (uniform) real distribution between min/mean and max/stdev
  rng(DType a,DType b) : m_dist(a,b) {}
};

/** Multiply entries of distributed matrix  with
 * a multiplier generated according to bernoulli_distribution
 */
template <typename DType=DataType>
void rng_bernoulli(const float p, DistMat* m) {

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
