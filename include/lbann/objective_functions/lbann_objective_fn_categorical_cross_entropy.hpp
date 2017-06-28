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

#ifndef LBANN_OBJECTIVE_FN_CATEGORICAL_CROSS_ENTROPY_HPP_INCLUDED
#define LBANN_OBJECTIVE_FN_CATEGORICAL_CROSS_ENTROPY_HPP_INCLUDED

#include "lbann/objective_functions/lbann_objective_fn.hpp"

namespace lbann {

namespace objective_functions {

class categorical_cross_entropy : public objective_fn {
 public:
  categorical_cross_entropy(lbann_comm *comm);
  categorical_cross_entropy(const categorical_cross_entropy& other) = default;
  categorical_cross_entropy& operator=(
    const categorical_cross_entropy& other) = default;
  ~categorical_cross_entropy();

  void setup(int num_neurons, int mini_batch_size);
  void fp_set_std_matrix_view(int cur_mini_batch_size);
  double compute_categorical_cross_entropy(const AbsDistMat& predictions_v,
                                           const AbsDistMat& groundtruth_v);
  double compute_obj_fn(const AbsDistMat& predictions_v,
                        const AbsDistMat& groundtruth_v);
  void compute_obj_fn_derivative(const Layer& prev_layer,
                                 const AbsDistMat& predictions_v,
                                 const AbsDistMat& groundtruth_v,
                                 AbsDistMat& error_signal_v);
  std::string name() const { return "categorical cross entropy"; }
};

}  // namespace objective_functions

}  // namespace lbann

#endif  // LBANN_OBJECTIVE_FN_CATEGORICAL_CROSS_ENTROPY_HPP_INCLUDED
