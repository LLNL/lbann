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

#include "lbann/objective_functions/weight_regularization/l1.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

void l1_weight_regularization::setup(model& m) {
  objective_function_term::setup(m);

  // Check that term has no layer pointers
  if (!m_layers.empty()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to setup group lasso weight regularization with layer pointers";
    throw lbann_exception(err.str());
  }

  // Add all weights in model if no weights pointers are provided
  if (m_weights.empty()) {
    for (weights* w : m.get_weights()) {
      if (w->get_optimizer() != nullptr) {
        m_weights.push_back(w);
      }
    }
  }

}

void l1_weight_regularization::start_evaluation() {

}

EvalType l1_weight_regularization::finish_evaluation() {
  if (m_scale_factor == EvalType(0)) { return EvalType(0); }
  EvalType value = EvalType(0);
  for (weights* w : m_weights) {

    // Get matrices
    const AbsDistMat& values = w->get_values();
    const Mat& values_local = values.LockedMatrix();
    const int local_height = values_local.Height();
    const int local_width = values_local.Width();

    // Compute L1 regularization term
    EvalType sum = 0;
#pragma omp taskloop collapse(2) default(shared) /// @todo reduction(+:sum)
    for (int col = 0; col < local_width; ++col) {
      for (int row = 0; row < local_height; ++row) {
        const EvalType val = values_local(row, col);
        #pragma omp critical
        sum += val >= EvalType(0) ? val : - val;
      }
    }
    value += get_comm().allreduce(sum, values.DistComm());

  }
  return m_scale_factor * value;
}

void l1_weight_regularization::compute_weight_regularization() {
  if (m_scale_factor == EvalType(0)) { return; }
  AbsDistMat* gradient;
  for (weights* w : m_weights) {

    // Get matrices
    const AbsDistMat& values = w->get_values();
    const Mat& values_local = values.LockedMatrix();
    const int local_height = values_local.Height();
    const int local_width = values_local.Width();

    // Compute gradient
    gradient = values.Copy();
    Mat& gradient_local = gradient->Matrix();
#pragma omp taskloop collapse(2) default(shared)
    for (int col = 0; col < local_width; ++col) {
      for (int row = 0; row < local_height; ++row) {
        const DataType val = values_local(row, col);
        DataType& grad = gradient_local(row, col);
        if (val > DataType(0)) {
          grad = DataType(1);
        } else if (val < DataType(0)) {
          grad = DataType(-1);
        } else {
          grad = DataType(0);
        }
      }
    }
    w->get_optimizer()->add_to_gradient(*gradient, m_scale_factor);
    delete gradient;

  }
}

}  // namespace lbann
