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

#ifndef SELU_HPP_INCLUDED
#define SELU_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/**
 * SELU: scaled exponential linear unit.
 * See: Klambauer et al. "Self-Normalizing Neural Networks", 2017.
 * https://arxiv.org/abs/1706.02515
 * By default, this assumes the goal is to normalize to 0 mean/unit variance.
 * To accomplish this, you should also normalize input to 0 mean/unit variance
 * (z-score), initialize with 0 mean, 1/n variance (He), and use the SELU
 * dropout.
 */
template <data_layout T_layout>
class selu_layer : public entrywise_activation_layer {
 public:
  selu_layer(int index, lbann_comm *comm,
             DataType alpha = DataType(1.6732632423543772848170429916717),
             DataType scale = DataType(1.0507009873554804934193349852946)) :
    entrywise_activation_layer(index, comm),
    m_alpha(alpha), m_scale(scale)
  {
    initialize_distributed_matrices();
  }

  selu_layer* copy() const { return new selu_layer(*this); }

  std::string get_type() const { return "SELU"; }

  virtual inline void initialize_distributed_matrices() {
    entrywise_activation_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

 protected:
  DataType activation_function(DataType z) {
    return (z >= DataType(0)) ? m_scale*z : m_scale*(m_alpha*std::expm1(z));
  }
  DataType activation_function_gradient(DataType z) {
    return (z >= DataType(0)) ? m_scale : m_scale*m_alpha*std::exp(z);
  }
 private:
  /** Alpha parameter for the ELU. */
  DataType m_alpha;
  /** Scaling parameter for the result of the ELU. */
  DataType m_scale;
};

}  // namespace lbann

#endif  // SELU_HPP_INCLUDED
