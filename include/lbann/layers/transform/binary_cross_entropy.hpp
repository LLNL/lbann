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

#ifndef LBANN_LAYER_BINARY_CROSS_ENTROPY_HPP_INCLUDED
#define LBANN_LAYER_BINARY_CROSS_ENTROPY_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Binary cross entropy loss function as a layer.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class binary_cross_entropy_layer : public transform_layer {
 public:

  binary_cross_entropy_layer(lbann_comm *comm,
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

  binary_cross_entropy_layer* copy() const override { return new binary_cross_entropy_layer(*this); }
  std::string get_type() const override { return "Binary cross entropy layer"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " BCE layer ; parents: ";
    for (size_t i=0; i<this->m_parent_layers.size(); i++) {
      s << this->m_parent_layers[i]->get_name() << " " << this->m_parent_layers[i]->get_type() << " ";
    }
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  protected:
void fp_compute() {
  //@todo: check that the order is not flipped 
  const auto& predictions = get_prev_activations(0);
  const auto& ground_truth = get_prev_activations(1);
  auto& output_local = get_activations().Matrix();
  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  
  // Matrix parameters
  const int local_height = predictions_local.Height();
  const int local_width = predictions_local.Width();

  // Compute sum of cross entropy terms
  #pragma omp parallel for collapse(2)
  for (int col = 0; col < local_width; ++col) {
    for (int row = 0; row < local_height; ++row) {
      const DataType true_val = ground_truth_local(row, col);
      const DataType pred_val = predictions_local(row, col);
      //@todo: add check when the same implementation in objective function is deprecated
      /*#ifdef LBANN_DEBUG
      binary_cross_entropy_debug::check_entry(ground_truth.GlobalRow(row),
                                              ground_truth.GlobalCol(col),
                                              true_val,
                                              pred_val);
      #endif // LBANN_DEBUG*/
      DataType sum = DataType(0);
      if (true_val > DataType(0)) {
        sum += - true_val * std::log(pred_val);
      }
      if (true_val < DataType(1)) {
        sum += - (EvalType(1) - true_val) * std::log(EvalType(1) - pred_val);
      }
      output_local(row,col) = sum;
    }
  }

}

void bp_compute() {

  //@todo: check that the order is not flipped 
  const auto& predictions = get_prev_activations(0);
  const auto& ground_truth = get_prev_activations(1);
  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  auto& gradient_local = get_error_signals().Matrix();

  // Matrix parameters
  const El::Int local_height = gradient_local.Height();
  const El::Int local_width = gradient_local.Width();

  // Compute gradient
  #pragma omp parallel for collapse(2)
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const DataType true_val = ground_truth_local(row, col);
      const DataType pred_val = predictions_local(row, col);
      /*#ifdef LBANN_DEBUG
      binary_cross_entropy_debug::check_entry(ground_truth.GlobalRow(row),
                                              ground_truth.GlobalCol(col),
                                              true_val,
                                              pred_val);
      #endif // LBANN_DEBUG*/
      DataType grad_val = DataType(0);
      if (true_val > DataType(0)) {
        grad_val += - true_val / pred_val;
      }
      if (true_val < DataType(1)) {
        grad_val += (DataType(1) - true_val) / (DataType(1) - pred_val);
      }
      gradient_local(row, col) = grad_val;
    }
  }
 
  //Add prev error signal? /evaluate layer--- all ones 
  /*El::Axpy(DataType(1), get_prev_error_signals(),
               get_error_signals());*/

}


};

} // namespace lbann

#endif // LBANN_LAYER_BINARY_CROSS_ENTROPY_HPP_INCLUDED
