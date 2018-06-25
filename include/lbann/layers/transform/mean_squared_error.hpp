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

#ifndef LBANN_LAYER_MSE_HPP_INCLUDED
#define LBANN_LAYER_MSE_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Mean squared error loss function as a layer.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class mean_squared_error_layer : public transform_layer {
 public:

  mean_squared_error_layer(lbann_comm *comm,
                 cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm) {

    // 2 parents expected: ground truth/label and predictions
    m_expected_num_parent_layers = 2;

  #ifdef LBANN_HAS_CUDNN
    // Activate GPU if needed
    if (cudnn != nullptr) {
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  mean_squared_error_layer* copy() const override { return new mean_squared_error_layer(*this); }
  std::string get_type() const override { return "Binary cross entropy layer"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " MSE layer ; parents: ";
    for (size_t i=0; i<this->m_parent_layers.size(); i++) {
      s << this->m_parent_layers[i]->get_name() << " " << this->m_parent_layers[i]->get_type() << " ";
    }
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  protected:
void fp_compute() {
  const auto& predictions = get_prev_activations(0);
  const auto& ground_truth = get_prev_activations(1);
  auto& output_local = get_activations().Matrix();
  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  
  // Matrix parameters
  const int local_height = predictions_local.Height();
  const int local_width = predictions_local.Width();

  // Compute squared sum of errors
  #pragma omp parallel for collapse(2)
  for (int col = 0; col < local_width; ++col) {
    for (int row = 0; row < local_height; ++row) {
      const DataType true_val = ground_truth_local(row, col);
      const DataType pred_val = predictions_local(row, col);
      const DataType error = true_val - pred_val;
      output_local(row,col) += error * error;
    }
  }

}

void bp_compute() {

  const auto& predictions = get_prev_activations(0);
  const auto& ground_truth = get_prev_activations(1);
  auto& gradient = get_error_signals();
  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  auto& gradient_local = gradient.Matrix();

  // Matrix parameters
  const int height = gradient.Height();
  const El::Int local_height = gradient_local.Height();
  const El::Int local_width = gradient_local.Width();

  // Compute gradient
  const DataType scale = DataType(2) / height; 
  #pragma omp parallel for collapse(2)
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const DataType true_val = ground_truth_local(row, col);
      const DataType pred_val = predictions_local(row, col);
      gradient_local(row, col) = (pred_val - true_val) * scale;
    }
  }
 
}


};

} // namespace lbann

#endif // LBANN_LAYER_MSE_HPP_INCLUDED
