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
// weights_initializer .hpp .cpp - Weights initializer classes
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_WEIGHTS_INITIALIZER_HPP
#define LBANN_WEIGHTS_INITIALIZER_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

namespace lbann {

/** Abstract weights initializer. */
class weights_initializer {
 public:

  /** Constructor. */
  weights_initializer(lbann_comm* comm)
    : m_comm(comm) {}

  /** Create a copy. */
  virtual weights_initializer* copy() const = 0;

  /** Construct a weights matrix with the initialization scheme.
   *  The caller is responsible for deallocating the matrix.
   */
  AbsDistMat* construct_matrix(int height = 0,
                               int width = 0,
                               El::Distribution col_dist = El::STAR,
                               El::Distribution row_dist = El::STAR) const;

  /** Initialize entries in a weights matrix. */
  virtual void intialize_entries(AbsDistMat& weights_matrix) const = 0;

 protected:

  /** LBANN communicator. */
  lbann_comm* m_comm;

};

/** Constant weights initializer. */
class constant_initializer : public weights_initializer {
 public:

  /** Constructor. */
  constant_initializer(lbann_comm* comm) 
    : weights_initializer(comm), m_value(value) {}

  /** Create a copy. */
  constant_initializer* copy() const override {
    return new constant_initializer(*this);
  }
  
  /** Initialize entries in a weights matrix to a constant value. */
  void intialize_entries(AbsDistMat& weights_matrix) const override;

 private:

  /** Constant value. */
  DataType m_value;

};

/** Uniform random weights initializer. */
class uniform_initializer : public weights_initializer {
 public:

  /** Constructor. */
  uniform_initializer(lbann_comm* comm,
                      DataType min_value = DataType(0),
                      DataType max_value = DataType(1))
    : weights_initializer(comm),
      m_min_value(min_value),
      m_max_value(max_value) {}

  /** Create a copy. */
  uniform_initializer* copy() const override {
    return new uniform_initializer(*this);
  }
  
  /** Draw weights matrix entries from uniform distribution. */
  void intialize_entries(AbsDistMat& weights_matrix) const override;

 private:

  /** Minimum value in uniform distribution. */
  DataType m_min_value;
  /** Maximum value in uniform distribution. */
  DataType m_max_value;

};

/** Normal random weights initializer. */
class normal_initializer : public weights_initializer {
 public:

  /** Constructor. */
  normal_initializer(lbann_comm* comm,
                     DataType mean = DataType(0),
                     DataType standard_deviation = DataType(1))
    : weights_initializer(comm),
      m_mean(mean),
      m_standard_deviation(standard_deviation) {}

  /** Create a copy. */
  normal_initializer* copy() const override {
    return new normal_initializer(*this);
  }

  /** Draw weights matrix entries from normal distribution. */
  void intialize_entries(AbsDistMat& weights_matrix) const override;

 private:

  /** Mean. */
  DataType m_mean;
  /** Standard deviation. */
  DataType m_standard_deviation;

};

} // namespace lbann

#endif // LBANN_WEIGHTS_INITIALIZER_HPP
