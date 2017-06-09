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
// lbann_layer_activations .hpp .cpp - Basic activations: sigmoid, tanh, reLU
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/lbann_layer_activations.hpp"
#include "lbann/utils/lbann_exception.hpp"

using namespace El;

namespace lbann {

void Activation::forwardProp(ElMat& m) {
  EntrywiseMap(m, std::function<DataType(const DataType&)>(
  [this] (const DataType& z) {
    return act(z);
  }));
}

void Activation::backwardProp(ElMat& m) {
  EntrywiseMap(m, std::function<DataType(const DataType&)>(
  [this] (const DataType& z) {
    return act_prime(z);
  }));
}

void Activation::backwardPropError(const ElMat& m, ElMat& prev_error_signal) {
  const Int height = m.LocalHeight();
  const Int width = m.LocalWidth();
  const Int m_LDim = m.LDim();
  const DataType *m_buf = m.LockedBuffer();
  const Int p_LDim = prev_error_signal.LDim();
  DataType *p_buf = prev_error_signal.Buffer();

  if (height == m_LDim && height == p_LDim) {
    // Contiguous memory.
    #pragma omp parallel for
    for (Int i = 0; i < height * width; ++i) {
      p_buf[i] = act_prime(m_buf[i]) * p_buf[i];
    }
  } else {
    // Non-contiguous.
    #pragma omp parallel for collapse(2)
    for (Int j = 0; j < width; ++j) {
      for (Int i = 0; i < height; ++i) {
        p_buf[i+j*p_LDim] = act_prime(m_buf[i+j*m_LDim]) * p_buf[i+j*p_LDim];
      }
    }
  }
}

const std::string Activation::activation_name(activation_type id) {
  switch(id) {
  case activation_type::SIGMOID:
    return "sigmoid";
    break;
  case activation_type::TANH:
    return "tanh";
    break;
  case activation_type::RELU:
    return "relu";
    break;
  case activation_type::ID:
    return "id";
    break;
  case activation_type::LEAKY_RELU:
    return "leaky_relu";
    break;
  case activation_type::SOFTPLUS:
    return "softplus";
    break;
  case activation_type::SMOOTH_RELU:
    return "smooth_relu";
    break;
  case activation_type::ELU:
    return "elu";
    break;
  default:
    throw lbann_exception("unknown activation_type");
  }
}

}  // namespace lbann
