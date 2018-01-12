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

#ifndef BENT_IDENTITY_HPP_INCLUDED
#define BENT_IDENTITY_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/** Bent identity activation function.
 *  See https://en.wikipedia.org/wiki/Bent_Identity_function
 */
template <data_layout T_layout>
class bent_identity_layer : public entrywise_activation_layer {
 public:
  bent_identity_layer(lbann_comm *comm)
    : entrywise_activation_layer(comm) {}
  bent_identity_layer* copy() const override { return new bent_identity_layer(*this); }

  std::string get_type() const override { return "bent identity"; }
  data_layout get_data_layout() const override { return T_layout; }

 protected:
  DataType activation(DataType z) const override {
    const DataType one = DataType(1);
    return (std::sqrt(z*z + one) - one) / 2 + z;
  }
  DataType activation_derivative(DataType z) const override {
    const DataType one = DataType(1);
    return z / (2 * std::sqrt(z*z + one)) + one;
  }
};

}  // namespace lbann

#endif  // BENT_IDENTITY_HPP_INCLUDED
