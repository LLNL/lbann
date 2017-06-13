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

#ifndef ACTIVATIONS_HPP_INCLUDED
#define ACTIVATIONS_HPP_INCLUDED

#include "lbann/utils/lbann_exception.hpp"
//#include "lbann/layers/lbann_layer.hpp"


namespace lbann {

/** Represent the type of activation function. */
enum class activation_type {
  //if you add or change the following enums, please also edit the
  //activation_name() method in the activation class
  SIGMOID = 1,
  TANH,
  RELU,
  ID,
  LEAKY_RELU,
  SOFTPLUS,
  SMOOTH_RELU,
  ELU
};

/** Base activation function class. */
template <class T_layout>
//class activation : public Layer {
class activation {
 public:
  //@todo: call Layer ctor
  activation() {}
  virtual ~activation() {}

  /** Apply the activation function elementwise to m. */
  virtual void forwardProp(ElMat& m) {
    EntrywiseMap(m, std::function<DataType(const DataType&)>(
    [this] (const DataType& z) {
      return act(z);
    }));
  }

  /** Apply the activation derivative function elementwise to m. */
  virtual void backwardProp(ElMat& m) {
    EntrywiseMap(m, std::function<DataType(const DataType&)>(
    [this] (const DataType& z) {
      return act_prime(z);
    }));
  }

  /**
   * Apply the activation derivative function and then multiply by the error
   * signal in one step, storing into prev_error_signal.
   */
  virtual void backwardPropError(const ElMat& m, ElMat& prev_error_signal) {
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

  static const std::string activation_name(activation_type id) {
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

 protected:
  /** The activation function. */
  virtual DataType act(const DataType& z) = 0;
  /** The derivative of the activation function. */
  virtual DataType act_prime(const DataType& z) = 0;
};


}  // namespace lbann

#endif  // ACTIVATIONS_HPP_INCLUDED
