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

#ifndef POWER_HPP_INCLUDED
#define POWER_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/** Power function. */
template <data_layout T_layout>
class power_layer : public entrywise_activation_layer {
 public:
  power_layer(lbann_comm *comm, EvalType exponent)
    : entrywise_activation_layer(comm), m_exponent(exponent) {}
  power_layer* copy() const override { return new power_layer(*this); }
  std::string get_type() const override { return "power"; }
  data_layout get_data_layout() const override { return T_layout; }

 protected:
  DataType activation(DataType z) const override {
    if (m_exponent == EvalType(2)) {
      return z * z;
    } else {
      return std::pow(z, m_exponent);
    }
  }
  DataType activation_derivative(DataType z) const override {
    if (m_exponent == EvalType(2)) {
      return 2 * z;
    } else {
      return m_exponent * std::pow(z, m_exponent - EvalType(1));
    }
  }

 private:

  /** Exponent for power function. */
  const EvalType m_exponent;

};

} // namespace lbann

#endif // POWER_HPP_INCLUDED
