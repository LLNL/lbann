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

#include "lbann/objective_functions/cross_entropy.hpp"
#include "lbann/layers/activations/softmax.hpp"
#include <typeinfo>
#include <typeindex>
#include <limits>

namespace lbann {

namespace objective_functions {

cross_entropy::cross_entropy(bool categorical_ground_truth)
  : objective_function(),
    m_categorical_ground_truth(categorical_ground_truth) {}

void cross_entropy::setup(const Layer& prev_layer) {

  // Activate softmax-cross-entropy shortcut if possible
  if(m_categorical_ground_truth) {
    const std::type_info& prev_layer_type = typeid(prev_layer);
    const std::type_info& data_parallel_softmax_type
      = typeid(softmax_layer<data_layout::DATA_PARALLEL>);
    const std::type_info& model_parallel_softmax_type
      = typeid(softmax_layer<data_layout::MODEL_PARALLEL>);
    if((std::type_index(prev_layer_type)
        == std::type_index(data_parallel_softmax_type))
       || (std::type_index(prev_layer_type)
           == std::type_index(model_parallel_softmax_type))) {
      m_shortcut_softmax_layer = &prev_layer;
    }
  }

}

void cross_entropy::compute_value(const AbsDistMat& predictions,
                                  const AbsDistMat& ground_truth) {

  // Get local matrices and matrix parameters
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  const El::Int width = predictions.Width();
  const El::Int local_height = predictions_local.Height();
  const El::Int local_width = predictions_local.Width();

  // Compute sum of cross entropy terms with Kahan summation
  double sum = 0;
  double correction = 0;
  for(El::Int col = 0; col < local_width; ++col) {
    for(El::Int row = 0; row < local_height; ++row) {
      const double true_val = ground_truth_local(row, col);
      const double pred_val = predictions_local(row, col);
      double term = - true_val * std::log(pred_val);
      term += correction;
      const double next_sum = sum + term;
      correction = term - (next_sum - sum);
      sum = next_sum;
    }
  }

  // Compute mean cross entropy across mini-batch
  double mean_cross_entropy = sum / width;
  mean_cross_entropy = El::mpi::AllReduce(mean_cross_entropy,
                                          predictions.DistComm());

  // Update objective function value
  add_to_value(mean_cross_entropy);

}

void cross_entropy::compute_gradient(const AbsDistMat& predictions,
                                     const AbsDistMat& ground_truth,
                                     AbsDistMat& gradient) {

  // Apply softmax-cross-entropy shortcut if activated
  if(m_shortcut_softmax_layer != nullptr) {
    El::LockedView(gradient, ground_truth);
    return;
  }

  // Get local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  Mat& gradient_local = gradient.Matrix();
  
  // Compute gradient
  El::IndexDependentFill(gradient_local,
                         (std::function<DataType(El::Int,El::Int)>)
                         ([&predictions_local, &ground_truth_local]
                          (El::Int r, El::Int c) -> DataType {
                           const DataType true_val = ground_truth_local(r,c);
                           if(true_val != DataType(0)) {
                             const DataType pred_val = predictions_local(r,c);
                             return - true_val / pred_val;
                           }
                           else {
                             return DataType(0);
                           }
                         }));

}

}  // namespace objective_functions

}  // namespace lbann
