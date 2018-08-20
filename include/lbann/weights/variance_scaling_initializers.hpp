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

#ifndef LBANN_WEIGHTS_VARIANCE_SCALING_INITIALIZER_HPP
#define LBANN_WEIGHTS_VARIANCE_SCALING_INITIALIZER_HPP

#include "lbann/weights/initializer.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** Abstract variance scaling weights initializer.
 *  Values are randomly sampled from a probability distribution with a
 *  variance determined by a "fan-in" and a "fan-out" parameter. This
 *  is a generalization of "Xavier" initialization.
 */
class variance_scaling_initializer : public weights_initializer {
public:
  variance_scaling_initializer(probability_distribution dist);
  void fill(AbsDistMat& matrix) override;
  
  /** Set fan-in parameter. */
  void set_fan_in(IntType fan_in) { m_fan_in = fan_in; }
  /** Set fan-out parameter. */
  void set_fan_out(IntType fan_out) { m_fan_out = fan_out; }

protected:
  /** Get probability distribution variance. */
  virtual DataType get_variance(IntType fan_in, IntType fan_out) = 0;
  
private:
  /** Probability distribution. */
  probability_distribution m_prob_dist;
  /** Fan-in parameter. */
  IntType m_fan_in;
  /** Fan-out parameter.*/
  IntType m_fan_out;

};

/** Glorot initializer.
 *  Initialize with variance of 2 / (fan-in + fan-out). Also called
 *  Xavier initialization.
 */
class glorot_initializer : public variance_scaling_initializer {
public:
  glorot_initializer(probability_distribution prob_dist)
    : variance_scaling_initializer(prob_dist) {}
  glorot_initializer* copy() const override {
    return new glorot_initializer(*this);
  }
protected:
  DataType get_variance(IntType fan_in, IntType fan_out) override;
};

/** He initializer.
 *  Initialize with variance of 2 / fan-in.
 */
class he_initializer : public variance_scaling_initializer {
public:
  he_initializer(probability_distribution prob_dist)
    : variance_scaling_initializer(prob_dist) {}
  he_initializer* copy() const override {
    return new he_initializer(*this);
  }
protected:
  DataType get_variance(IntType fan_in, IntType fan_out) override;
};

/** LeCun initializer.
 *  Initialize with variance of 1 / fan-in.
 */
class lecun_initializer : public variance_scaling_initializer {
public:
  lecun_initializer(probability_distribution prob_dist)
    : variance_scaling_initializer(prob_dist) {}
  lecun_initializer* copy() const override {
    return new lecun_initializer(*this);
  }
protected:
  DataType get_variance(IntType fan_in, IntType fan_out) override;
};

} // namespace lbann

#endif // LBANN_WEIGHTS_VARIANCE_SCALING_INITIALIZER_HPP
