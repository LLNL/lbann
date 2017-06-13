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

#ifndef CREATE_ACTIVATION_HPP_INCLUDED
#define CREATE_ACTIVATION_HPP_INCLUDED

#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/id.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/activations/sigmoid.hpp"
#include "lbann/layers/activations/smooth_relu.hpp"
#include "lbann/layers/activations/tanh.hpp"
#include "lbann/layers/activations/softplus.hpp"

#include "lbann/utils/lbann_exception.hpp"

namespace lbann {

/** Return a new Activation class of type act_fn. */
template<typename... Args>
activation<data_layout> *new_activation(activation_type act_fn, Args... params) {
  switch (act_fn) {
  case activation_type::SIGMOID:
    return new sigmoid_layer<data_layout>();
  case activation_type::TANH:
    return new tanh_layer<data_layout>();
  case activation_type::RELU:
    return new relu_layer<data_layout>();
  case activation_type::ID:
    return new id_layer<data_layout>();
  case activation_type::LEAKY_RELU:
    return new leaky_relu_layer<data_layout>(params...);
  case activation_type::SOFTPLUS:
    return new softplus_layer<data_layout>();
  case activation_type::SMOOTH_RELU:
    return new smooth_relu_layer<data_layout>();
  case activation_type::ELU:
    return new elu_layer<data_layout>(params...);
  default:
    throw lbann_exception("Unsupported activation type.");
  }
  return nullptr;  // Never reached.
}

}  // namespace lbann

#endif  // ACTIVATIONS_HPP_INCLUDED
