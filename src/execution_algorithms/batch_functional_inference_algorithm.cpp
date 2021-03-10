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

#include "lbann/execution_algorithms/batch_functional_inference_algorithm.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/callbacks/callback.hpp"

namespace lbann {

template <typename TensorDataType>
int batch_functional_inference_algorithm::
get_label(El::AbstractDistMatrix<TensorDataType> const& label_data, int row) {
  TensorDataType max = 0;
  int idx = 0;
  TensorDataType col_value;
  int col_count = label_data.Height();
  for (int i = 0; i < col_count; i++) {
    col_value = label_data.Get(row, i);
    if (col_value > max) {
      max = col_value;
      idx = i;
    }
  }
  return idx;
}

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>
batch_functional_inference_algorithm::
infer(model& model,
      El::AbstractDistMatrix<TensorDataType> const& samples,
      std::string output_layer,
      size_t mbs) {
  size_t samples_size = samples.Height();
  El::AbstractDistMatrix<TensorDataType> labels(samples_size, 1);

  // Infer on mini batches
  for (size_t i = 0; i < samples_size; i+=mbs) {
    size_t mbs_idx = std::min(i+mbs, samples_size);
    auto mini_batch_samples = El::View(samples, El::IR(i, mbs_idx), El::ALL);
    auto& mbl = infer_mini_batch(model, mini_batch_samples, output_layer);

    // Fill labels, right now this assumes a softmax output for a
    // classification problem
    for (size_t j = i; j < mbs_idx; j++) {
      // This probably doesn't work for a distributed matrix and will be
      // changed when I properly test it with an external driver application
      labels[j] = get_label(mbl, j-i);
    }
  }

  return labels;
}

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>
batch_functional_inference_algorithm::
infer_mini_batch(model& model,
                 El::AbstractDistMatrix<TensorDataType> const& samples,
                 std::string output_layer) {
  // Insert samples into input layer here
  for (int i=0; i < model.get_num_layers(); i++) {
    auto& l = model.get_layer(i);
    if (l.get_type() == "input") {
      auto& il = dynamic_cast<input_layer<DataType>&>(l);
      il.set_samples(samples);
    }
  }

  model.forward_prop(execution_mode::inference);

  // Get inference labels
  // Currently this just gets the output tensor of size sample_n X label_n
  // We will need to work out how to process the output to give the correct
  // values for different models (e.g., classification vs regression)
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
