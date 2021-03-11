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
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/callbacks/callback.hpp"

namespace lbann {

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>
batch_functional_inference_algorithm::
infer(model& model,
      El::AbstractDistMatrix<TensorDataType> const& samples,
      std::string output_layer,
      size_t mbs) {
  // Matrix for collecting mini batch labels
  El::AbstractDistMatrix<TensorDataType> labels(samples.Height(), 1);

  for (size_t i = 0; i < mbs; i++) {
    // Need to view subsamples for mini batch
    auto& mbl = infer_mini_batch(model, samples, output_layer);
    // Will need a method for concatenating returned matrices here
  }
  return labels;
}

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>
batch_functional_inference_algorithm::
infer_mini_batch(model& model,
                 El::AbstractDistMatrix<TensorDataType> const& samples,
                 std::string output_layer) {
  // TODO: Insert samples into input layer here
  model.forward_prop(execution_mode::inference);

  // Get inference labels
  El::AbstractDistMatrix<TensorDataType> labels;
  for (const auto* l : model.get_layers()) {
    if (l->get_name() == output_layer) {
      auto const& dtl = dynamic_cast<lbann::data_type_layer<float> const&>(*l);
      labels = dtl.get_activations();
    }
  }

  return labels;
}

}  // namespace lbann
