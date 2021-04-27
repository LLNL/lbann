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

#ifndef LBANN_BATCH_INFERENCE_ALGORITHM_HPP
#define LBANN_BATCH_INFERENCE_ALGORITHM_HPP

#include "lbann/models/model.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/io/input_layer.hpp"
#include "lbann/callbacks/callback.hpp"


namespace lbann {

/** @brief Class for LBANN batch inference algorithms. */
class batch_functional_inference_algorithm {
public:

  /** Constructor. */
  batch_functional_inference_algorithm() {};
  /** Copy constructor. */
  batch_functional_inference_algorithm(const batch_functional_inference_algorithm& other) = default;
  /** Copy assignment operator. */
  batch_functional_inference_algorithm& operator=(const batch_functional_inference_algorithm& other) = default;
  /** Move constructor. */
  batch_functional_inference_algorithm(batch_functional_inference_algorithm&& other) = default;
  /** Move assignment operator. */
  batch_functional_inference_algorithm& operator=(batch_functional_inference_algorithm&& other) = default;
  /** Destructor. */
  virtual ~batch_functional_inference_algorithm() = default;
  /** Copy training_algorithm. */
  //  virtual batch_functional_inference_algorithm* copy() const = default;

  std::string get_name() const { return "batch_functional_inference"; }

  // ===========================================
  // Execution
  // ===========================================

  /** Infer on samples from a data coordinator with a given model. */
  template <typename DataT, El::Dist CDist, El::Dist RDist, El::DistWrap DistView, El::Device Device>
  El::Matrix<int, El::Device::CPU>
  infer(observer_ptr<model> model,
        El::DistMatrix<DataT, CDist, RDist, DistView, Device> const& samples,
        std::string output_layer,
        size_t mbs=0) {
    size_t samples_size = samples.Height();
    El::Matrix<int, El::Device::CPU> labels(samples_size, 1);

    // Infer on mini batches
    for (size_t i = 0; i < samples_size; i+=mbs) {
      size_t mbs_idx = std::min(i+mbs, samples_size);
      //El::DistMatrix<DataT, CDist, RDist, DistView, Device> mini_batch_samples(mbs,128*128);
      auto mini_batch_samples = El::LockedView(samples, El::IR(i, mbs_idx), El::ALL);
      auto mbl = infer_mini_batch(*model, mini_batch_samples, output_layer);

      // Fill labels, right now this assumes a softmax output for a
      // classification problem
      for (size_t j = i; j < mbs_idx; j++) {
        // This probably doesn't work for a distributed matrix and will be
        // changed when I properly test it with an external driver application
        //labels(j) = get_label(mbl, j-i);
      }
    }

    return labels;
  }


protected:
  /** return label for a given row of softmax output. */
  template <typename DataT, El::Dist CDist, El::Dist RDist, El::DistWrap DistView, El::Device Device>
  int get_label(El::DistMatrix<DataT, CDist, RDist, DistView, Device> & label_data, int row) {
    DataT max = 0;
    int idx = 0;
    DataT col_value;
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

  /** Infer on one mini batch with a given model. */
  template <typename DataT, El::Dist CDist, El::Dist RDist, El::DistWrap DistView, El::Device Device>
  const El::BaseDistMatrix*
  infer_mini_batch(model& model,
                   El::DistMatrix<DataT, CDist, RDist, DistView, Device> const& samples,
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
    const El::BaseDistMatrix *labels;
    for (const auto* l : model.get_layers()) {
      if (l->get_name() == output_layer) {
        auto const& dtl = dynamic_cast<lbann::data_type_layer<float> const&>(*l);
        labels = &dtl.get_activations();
      }
    }

    return labels;
}

};

}  // namespace lbann

#endif  // LBANN_BATCH_INFERENCE_ALGORITHM_HPP
