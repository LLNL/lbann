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

#ifndef LBANN_WEIGHTS_INITIALIZER_HPP
#define LBANN_WEIGHTS_INITIALIZER_HPP

#include "lbann/base.hpp"

namespace lbann {

/** Abstract weights initializer. */
class weights_initializer {
public:
  weights_initializer() = default;
  virtual ~weights_initializer() = default;

  /** Create a copy. */
  virtual weights_initializer* copy() const = 0;

  /** Initialize entries in a weights matrix. */
  virtual void fill(AbsDistMat& matrix) = 0;

};

/** Constant weights initializer. */
class constant_initializer : public weights_initializer {
public:
  constant_initializer(DataType value)
    : weights_initializer(), m_value(value) {}
  ~constant_initializer() override = default;
  constant_initializer* copy() const override {
    return new constant_initializer(*this);
  }
  void fill(AbsDistMat& matrix) override;

private:
  /** Constant value. */
  DataType m_value;

};

/** Uniform random weights initializer. */
class uniform_initializer : public weights_initializer {
 public:
  uniform_initializer(DataType min = DataType(0),
                      DataType max = DataType(1))
    : weights_initializer(), m_min(min), m_max(max) {}
  ~uniform_initializer() override = default;
  uniform_initializer* copy() const override {
    return new uniform_initializer(*this);
  }
  void fill(AbsDistMat& matrix) override;

private:
  /** Uniform distribution minimum. */
  DataType m_min;
  /** Uniform distribution maximum. */
  DataType m_max;

};

/** Normal random weights initializer. */
class normal_initializer : public weights_initializer {
public:
  normal_initializer(DataType mean = DataType(0),
                     DataType standard_deviation = DataType(1))
    : weights_initializer(),
      m_mean(mean),
      m_standard_deviation(standard_deviation) {}
  ~normal_initializer() override = default;
  normal_initializer* copy() const override {
    return new normal_initializer(*this);
  }
  void fill(AbsDistMat& matrix) override;

private:
  /** Mean. */
  DataType m_mean;
  /** Standard deviation. */
  DataType m_standard_deviation;

};

} // namespace lbann

#endif // LBANN_WEIGHTS_INITIALIZER_HPP
