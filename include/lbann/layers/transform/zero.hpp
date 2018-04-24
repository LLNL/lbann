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

#ifndef LBANN_LAYER_ZERO_HPP_INCLUDED
#define LBANN_LAYER_ZERO_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** Layer outputs (transform previous activations to) zeros.
 * use case: transforms part or all samples in a minibatch to zero
 * @param first_half output zeros for the first half of minibatch samples if true
 * @param second_half output zeros for second half of minibatch samples if true
 * @todo generalzie if there are other use cases
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class zero_layer : public transform_layer {
 private:
  bool  m_first_half;
  bool  m_second_half;

 public:
 zero_layer(lbann_comm *comm,
              bool first_half=true,
              bool second_half=true,
              cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm),
      m_first_half(first_half),
      m_second_half(second_half) {

  }
  zero_layer* copy() const override { return new zero_layer(*this); }
  std::string get_type() const override { return "zero"; }
  data_layout get_data_layout() const override { return T_layout; }

  /** Returns description of constructor params */
  std::string get_description() const override {
    std::stringstream s;
     s << "zero_layer  first half: " << m_first_half << "second half: " << m_second_half  
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
        if(m_first_half)
        y = input.GlobalCol(col) < local_width/2 ?  DataType(0) : x;
        if(m_second_half)
        y = input.GlobalCol(col) >= local_width/2 ?  DataType(0) : x;
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
        if(m_first_half)
        dx += input.GlobalCol(col) < local_width/2 ?  DataType(0) : dy;
        if(m_second_half)
        dx += input.GlobalCol(col) >= local_width/2 ?  DataType(0) : dy;
      }
    }
  }


};

} // namespace lbann

#endif // LBANN_LAYER_ZERO_HPP_INCLUDED
