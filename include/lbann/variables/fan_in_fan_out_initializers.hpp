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
// fan_in_fan_out_initializer .hpp .cpp - Fan-in-fan-out initializer classes
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_FAN_IN_FAN_OUT_INITIALIZER_HPP
#define LBANN_FAN_IN_FAN_OUT_INITIALIZER_HPP

#include "lbann/variables/initializer.hpp"

namespace lbann {

/** Abstract base class for fan-in-fan-out variable initializers.
 *  The initialization scheme only depends on the value of a fan-in
 *  dimension and a fan-out dimension.
 */
class fan_in_fan_out_initializer : public variable_initializer {
 public:

  /** Constructor. */
  fan_in_fan_out_initializer(lbann_comm* comm)
    : variable_initializer(comm), m_fan_in(0), m_fan_out(0) {}

  /** Set fan-in dimension. */
  void set_fan_in(int fan_in) { m_fan_in = fan_in; }
  /** Set fan-out dimension. */
  void set_fan_out(int fan_out) { m_fan_out = fan_out; }

 protected:

  /** Fan-in dimension. */
  int m_fan_in;
  /** Fan-out dimension.*/
  int m_fan_out;

};

/** Glorot normal initializer.
 *  Also called Xavier normal initialization.
 */
class glorot_normal_initializer : public fan_in_fan_out_initializer {
 public:

  /** Constructor. */
  glorot_normal_initializer(lbann_comm* comm) 
    : fan_in_fan_out_initializer(comm) {}
  
  /** Create a copy. */
  glorot_normal_initializer* copy() const override {
    return new glorot_normal_initializer(*this);
  }

  /** Initialize variable matrix entries. */
  void intialize_entries(AbsDistMat& variable_matrix) const override;

};

/** Glorot uniform initializer.
 *  Also called Xavier uniform initialization.
 */
class glorot_uniform_initializer : public fan_in_fan_out_initializer {
 public:

  /** Constructor. */
  glorot_uniform_initializer(lbann_comm* comm) 
    : fan_in_fan_out_initializer(comm) {}

  /** Create a copy. */
  glorot_uniform_initializer* copy() const override {
    return new glorot_uniform_initializer(*this);
  }
  
  /** Initialize variable matrix entries. */
  void intialize_entries(AbsDistMat& variable_matrix) const override;

};

/** He normal initializer. */
class he_normal_initializer : public fan_in_fan_out_initializer {
 public:

  /** Constructor. */
  he_normal_initializer(lbann_comm* comm) 
    : fan_in_fan_out_initializer(comm) {}

  /** Create a copy. */
  he_normal_initializer* copy() const override {
    return new he_normal_initializer(*this);
  }
  
  /** Initialize variable matrix entries. */
  void intialize_entries(AbsDistMat& variable_matrix) const override;

};


/** He uniform initializer. */
class he_uniform_initializer : public fan_in_fan_out_initializer {
 public:

  /** Constructor. */
  he_uniform_initializer(lbann_comm* comm) 
    : fan_in_fan_out_initializer(comm) {}

  /** Create a copy. */
  he_uniform_initializer* copy() const override {
    return new he_uniform_initializer(*this);
  }
  
  /** Initialize variable matrix entries. */
  void intialize_entries(AbsDistMat& variable_matrix) const override;

};

} // namespace lbann

#endif // LBANN_FAN_IN_FAN_OUT_INITIALIZER_HPP
