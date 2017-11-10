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
// variable_initializer .hpp .cpp - Variable initializer classes
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_VARIABLES_INITIALIZER_HPP
#define LBANN_VARIABLES_INITIALIZER_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

namespace lbann {

/** Abstract base class for variable initializers. */
class variable_initializer {
 public:

  /** Constructor. */
  variable_initializer(lbann_comm* comm)
    : m_comm(comm) {}

  /** Create a copy. */
  virtual variable_initializer* copy() const = 0;

  /** Construct a variable matrix with the initialization scheme.
   *  The caller is responsible for deallocating the matrix.
   */
  AbsDistMat* construct_matrix(int height = 0,
                               int width = 0,
                               El::Distribution col_dist = El::STAR,
                               El::Distribution row_dist = El::STAR) const;

  /** Initialize entries in a variable matrix. */
  virtual void intialize_entries(AbsDistMat& variable_matrix) const = 0;

 protected:

  /** LBANN communicator. */
  lbann_comm* m_comm;

};

/** Variable initializer that sets to a constant value. */
class constant_initializer : public variable_initializer {
 public:

  /** Constructor. */
  constant_initializer(lbann_comm* comm) 
    : variable_initializer(comm), m_value(value) {}

  /** Create a copy. */
  constant_initializer* copy() const override {
    return new constant_initializer(*this);
  }
  
  /** Initialize entries in a variable matrix to a constant value. */
  void intialize_entries(AbsDistMat& variable_matrix) const override;

 private:

  /** Constant value. */
  DataType m_value;

};

/** Variable initializer that draws from a uniform distribution. */
class uniform_initializer : public variable_initializer {
 public:

  /** Constructor. */
  uniform_initializer(lbann_comm* comm,
                      DataType min_value = DataType(0),
                      DataType max_value = DataType(1))
    : variable_initializer(comm),
      m_min_value(min_value),
      m_max_value(max_value) {}

  /** Create a copy. */
  uniform_initializer* copy() const override {
    return new uniform_initializer(*this);
  }
  
  /** Draw variable matrix entries from uniform distribution. */
  void intialize_entries(AbsDistMat& variable_matrix) const override;

 private:

  /** Minimum value in uniform distribution. */
  DataType m_min_value;
  /** Maximum value in uniform distribution. */
  DataType m_max_value;

};

/** Variable initializer that draws from a normal distribution. */
class normal_initializer : public variable_initializer {
 public:

  /** Constructor. */
  normal_initializer(lbann_comm* comm,
                     DataType mean = DataType(0),
                     DataType standard_deviation = DataType(1))
    : variable_initializer(comm),
      m_mean(mean),
      m_standard_deviation(standard_deviation) {}

  /** Create a copy. */
  normal_initializer* copy() const override {
    return new normal_initializer(*this);
  }

  /** Draw variable matrix entries from normal distribution. */
  void intialize_entries(AbsDistMat& variable_matrix) const override;

 private:

  /** Mean. */
  DataType m_mean;
  /** Standard deviation. */
  DataType m_standard_deviation;

};

} // namespace lbann

#endif // LBANN_VARIABLES_INITIALIZER_HPP
