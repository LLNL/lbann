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

#ifndef TANH_HPP_INCLUDED
#define TANH_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/** Hyperbolic tangent activation function. */
template <data_layout T_layout>
class tanh_layer : public entrywise_activation_layer {
 public:
  //tanh_layer(data_layout data_dist, uint index, lbann_comm *comm,
  tanh_layer(uint index, lbann_comm *comm,
             const uint mini_batch_size, uint num_neurons) :
    entrywise_activation_layer(index, comm,
                               mini_batch_size, num_neurons) { 
    set_name("tanh_layer");
    initialize_distributed_matrices(); 
    }

  virtual inline void initialize_distributed_matrices() {
    entrywise_activation_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual inline data_layout get_data_layout() { return T_layout; }

 protected:
  DataType activation_function(DataType z) {
    return std::tanh(z);
  }
  DataType activation_function_gradient(DataType z) {
    const DataType e = std::exp(DataType(2)*z);
    return (e - DataType(1)) / (e + DataType(1));
  }
};

}  // namespace lbann

#endif  // TANH_HPP_INCLUDED
