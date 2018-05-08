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

#ifndef LBANN_LAYER_BINARY_CROSS_ENTROPY_WITH_LOGITS_HPP_INCLUDED
#define LBANN_LAYER_BINARY_CROSS_ENTROPY_WITH_LOGITS_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** Compute logistic loss.
 * @param label - label for ground truth 0/1
 * @x - input activations, if x is coming from sigmoid layer, then this may be equivalent to:
 * sigmoid_cross_entropy_with_logits (https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)
 * if (label=1) loss = -log(x) 
 * if (label=0) loss = log(1-x)
 * @todo: GPU Implementation
 * @check that m_true_label is zero or 1
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class binary_cross_entropy_with_logits_layer : public transform_layer {
 private:
  int  m_true_label;

 public:
 binary_cross_entropy_with_logits_layer(lbann_comm *comm,
              int true_label,
              cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm),
      m_true_label(true_label)
      {  }
  binary_cross_entropy_with_logits_layer* copy() const override { return new binary_cross_entropy_with_logits_layer(*this); }
  std::string get_type() const override { return "binary_cross_entropy_with_logits"; }
  data_layout get_data_layout() const override { return T_layout; }

  /** Returns description of constructor params */
  std::string get_description() const override {
    std::stringstream s;
     s << "binary_cross_entropy_with_logits_layer  ground truth label: " << m_true_label  
       << " dataLayout: " << this->get_data_layout_string(get_data_layout());
     return s.str();
  }

 protected:

  void fp_compute() override {
    const auto& input = get_prev_activations();
    const auto& local_input = input.LockedMatrix();
    auto& local_output = get_local_activations();
    const int local_height = local_input.Height();
    const int local_width = local_input.Width();
    for (int col = 0; col < local_width; ++col) {
      for (int row = 0; row < local_height; ++row) {
        const DataType x = local_input(row, col);
        DataType& y = local_output(row, col);
        if(m_true_label)  y += -std::log(x); //if ground truth label is 1
        else y += -std::log(EvalType(1) - x); //if ground truth label is 0
      }
    }
  }

  void bp_compute() override {
    const auto& input = get_prev_error_signals();
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();
    const int local_height = input.LocalHeight();
    const int local_width = input.LocalWidth();
    for (int col = 0; col < local_width; ++col) {
      for (int row = 0; row < local_height; ++row) {
        const DataType dy = local_gradient_wrt_output(row, col);
        DataType& dx = local_gradient_wrt_input(row, col);
        if(m_true_label) dx += DataType(-1)/dy; //fudge value for dy?
        else dx += DataType(1) / (DataType(1) - dy);
      }
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_BINARY_CROSS_ENTROPY_WITH_LOGITS_HPP_INCLUDED
