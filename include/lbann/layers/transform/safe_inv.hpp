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

#ifndef LBANN_LAYER_SAFE_INV_HPP_INCLUDED
#define LBANN_LAYER_SAFE_INV_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Safe entrywise inversion (reciprocal).
 *  Output is zero if input is zero. See https://arxiv.org.abs/1606.06582
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class safe_inv_layer : public transform_layer {
 private:
  
  /** Threshhold for computing inverse. */
  DataType m_threshhold;

 public:

  safe_inv_layer(lbann_comm *comm,
                 DataType threshhold = DataType(0),
                 cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm), m_threshhold(threshhold) {}

  safe_inv_layer* copy() const override { return new safe_inv_layer(*this); }
  std::string get_type() const override { return "safe_inv"; }
  data_layout get_data_layout() const override { return T_layout; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
     s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
     return s.str();
  }

 protected:

  void fp_compute() override {
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();
    const int local_height = local_input.Height();
    const int local_width = local_input.Width();
    for (int col = 0; col < local_width; ++col) {
      for (int row = 0; row < local_height; ++row) {
        const DataType x = local_input(row, col);
        DataType& y = local_output(row, col);
        y = std::fabs(x) > m_threshhold ? 1 / x : DataType(0);
      }
    }
  }

  void bp_compute() override {
    const auto& local_input = get_local_prev_activations();
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();
    const int local_height = local_input.Height();
    const int local_width = local_input.Width();
    for (int col = 0; col < local_width; ++col) {
      for (int row = 0; row < local_height; ++row) {
        const DataType x = local_input(row, col);
        const DataType dy = local_gradient_wrt_output(row, col);
        DataType& dx = local_gradient_wrt_input(row, col);
        dx += std::fabs(x) > m_threshhold ?  - dy / (x * x) : DataType(0);
      }
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_SAFE_INV_HPP_INCLUDED
