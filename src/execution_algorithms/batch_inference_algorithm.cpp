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

#include "lbann/execution_algorithms/batch_inference_algorithm.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/data_coordinator/buffered_data_coordinator.hpp"
#include "lbann/models/model.hpp"
#include "lbann/callbacks/callback.hpp"

namespace lbann {

void batch_inference_algorithm::infer(model& model,
                                      data_coordinator& dc,
                                      size_t num_batches) {
  if (num_batches > 0) {
    for (size_t i = 0; i < num_batches; i++) { infer_mini_batch(model, dc); }
  } else {
    while (!infer_mini_batch(model, dc)) {}
  }
}

template <typename TensorDataType>
void batch_inference_algorithm::infer(model& model,
                                      El::AbstractDistMatrix<TensorDataType> const& samples,
                                      size_t num_batches) {
  // This code block will change, but depends on how samples inserted into the
  // input layer, for now this is ok...
  if (num_batches > 0) {
    for (size_t i = 0; i < num_batches; i++) { infer_mini_batch(model, samples); }
  } else {
    while (!infer_mini_batch(model, samples)) {}
  }
}

bool batch_inference_algorithm::infer_mini_batch(model& model,
                                                 data_coordinator& dc) {
  dc.fetch_data(execution_mode::inference);
  model.forward_prop(execution_mode::inference);
  const bool finished = dc.epoch_complete(execution_mode::inference);
  return finished;
}

template <typename TensorDataType>
bool batch_inference_algorithm::infer_mini_batch(model& model,
                                                 El::AbstractDistMatrix<TensorDataType> const& samples) {
  // TODO: Insert samples into input layer here
  model.forward_prop(execution_mode::inference);
  return true;
}

}  // namespace lbann
