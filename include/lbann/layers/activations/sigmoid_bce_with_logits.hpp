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

#ifndef SIGMOID_BCE_WITH_LOGITS_HPP_INCLUDED
#define SIGMOID_BCE_WITH_LOGITS_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/** Compute logistic loss.
 * @param label - label for ground truth 0/1
 * sigmoid_cross_entropy_with_logits (https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)
 * @check that m_true_label is zero or 1
 */
template <data_layout T_layout>
class sigmoid_bce_with_logits_layer : public entrywise_activation_layer {
 private:
  int  m_ground_truth_label;
 public:
  sigmoid_bce_with_logits_layer(lbann_comm *comm, int ground_truth_label) : entrywise_activation_layer(comm),
  m_ground_truth_label(ground_truth_label) {}
  sigmoid_bce_with_logits_layer* copy() const override { return new sigmoid_bce_with_logits_layer(*this); }
  std::string get_type() const override { return "sigmoid_bce_with_logits"; }
  data_layout get_data_layout() const override { return T_layout; }

 protected:
  DataType activation(DataType z) const override {
    DataType res;
    if( z>= 0) {
      DataType log_term = std::log(DataType(1) + std::exp(-z));
      if(m_ground_truth_label) res = log_term;
      else res = z + log_term;
    //for z < 0, reformulate to avoid overflow in exp(-z) 
    }else {
      DataType log_term = std::log(DataType(1) + std::exp(-(std::abs(z))));
      if(m_ground_truth_label) res = std::max(z,DataType(0)) - z + log_term;
      else res = std::max(z,DataType(0))  + log_term;
    }
    return res;
  }

  DataType activation_derivative(DataType z) const override {
    DataType res;
    if( z >=0 ) {
      DataType log_term_derivative = std::exp(-(z)) / (DataType(1) + std::exp(-(z)));
      if(m_ground_truth_label) res = DataType(0) - log_term_derivative;
      else res = DataType(1) - log_term_derivative;
    }else {
      DataType log_term_derivative = std::exp(-(std::abs(z))) / (DataType(1) + std::exp(-(std::abs(z))));
      if(m_ground_truth_label) res = log_term_derivative;
      else res = DataType(1) + log_term_derivative;
    }
    return res;
  }
};

} // namespace lbann

#endif // SIGMOID_BCE_WITH_LOGITS_HPP_INCLUDED
