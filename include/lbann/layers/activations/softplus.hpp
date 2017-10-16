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

#ifndef SOFTPLUS_HPP_INCLUDED
#define SOFTPLUS_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/**
 * Softplus activation function.
 * This is a smooth approximation of the ReLU.
 * See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
template <data_layout T_layout>
class softplus_layer : public entrywise_activation_layer {
public :
  softplus_layer(int index,
                 lbann_comm *comm) :
    entrywise_activation_layer(index, comm) { 
    initialize_distributed_matrices(); 
  }

  softplus_layer* copy() const { return new softplus_layer(*this); }

  std::string get_type() const { return "softplus"; }

  virtual inline void initialize_distributed_matrices() {
    entrywise_activation_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

 protected:
  DataType activation_function(DataType z) {
    // Warning: Not numerically stable.
    // Better approach is to determine a threshold so that for large z,
    // softplus(z) ~= z and for small z, softplus(z) ~= exp(z).
    return std::log1p(std::exp(z));
  }
  DataType activation_function_gradient(DataType z) {
    return DataType(1.0) / (DataType(1.0) + std::exp(-z));
  }
};

}  // namespace lbann

#endif  // SOFTPLUS_HPP_INCLUDED
