////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/callbacks/callback.hpp"
#include "lbann/data_ingestion/data_coordinator.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/io/input_layer.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

/** @brief Class for LBANN batch inference algorithms.
 *
 *  This execution algorithm is meant for running inference using a trained
 *  model and samples passed by the user from an external application.  The
 *  algorithm currently assumes that there is only 1 input layer in the model,
 *  and the output layer is a softmax layer.
 */
class batch_functional_inference_algorithm
{
public:
  /** Constructor. */
  batch_functional_inference_algorithm(){};
  /** Copy constructor. */
  batch_functional_inference_algorithm(
    const batch_functional_inference_algorithm& other) = default;
  /** Copy assignment operator. */
  batch_functional_inference_algorithm&
  operator=(const batch_functional_inference_algorithm& other) = default;
  /** Move constructor. */
  batch_functional_inference_algorithm(
    batch_functional_inference_algorithm&& other) = default;
  /** Move assignment operator. */
  batch_functional_inference_algorithm&
  operator=(batch_functional_inference_algorithm&& other) = default;
  /** Destructor. */
  virtual ~batch_functional_inference_algorithm() = default;

  std::string get_name() const { return "batch_functional_inference"; }

  std::string get_type() const { return "batch_functional_inference"; }

  // ===========================================
  // Execution
  // ===========================================

  /** @brief Run model inference on samples and return predicted categories.
   * @param[in] model A trained model
   * @param[in] samples A distributed matrix containing samples for model input
   * @param[in] mbs The max mini-batch size
   * @return Matrix of predicted labels (by index)
   */
  template <typename DataT,
            El::Dist CDist,
            El::Dist RDist,
            El::DistWrap DistView,
            El::Device Device>
  El::Matrix<int, El::Device::CPU>
  infer(observer_ptr<model> model,
        El::DistMatrix<DataT, CDist, RDist, DistView, Device> const& samples,
        size_t mbs)
  {
    if (mbs <= 0) {
      LBANN_ERROR("mini-batch size must be larger than 0");
    }

    // Make matrix for returning predicted labels
    size_t samples_size = samples.Height();
    El::Matrix<int, El::Device::CPU> labels(samples_size, 1);

    // BVE FIXME
    // Create an SGD_execution_context so that layer.forward_prop can get the
    // mini_batch_size - This should be fixed in the future, when SGD is not so
    // hard-coded into the model & layers
    auto c = SGDExecutionContext(execution_mode::inference);
    model->reset_mode(c, execution_mode::inference);
    // Explicitly set the size of the mini-batch that the model is executing
    model->set_current_mini_batch_size(mbs);

    // Infer on mini batches
    for (size_t i = 0; i < samples_size; i += mbs) {
      size_t mb_idx = std::min(i + mbs, samples_size);
      auto mb_range = El::IR(i, mb_idx);
      auto mb_samples = El::LockedView(samples, mb_range, El::ALL);
      auto mb_labels = El::View(labels, mb_range, El::ALL);

      infer_mini_batch(*model, mb_samples);
      get_labels(*model, mb_labels);
    }

    return labels;
  }

protected:
  /** @brief Run model inference on a single mini-batch of samples
   * This method takes a mini-batch of samples, inserts them into the input
   * layer of the model, and runs forward prop on the model.
   * @param[in] model A trained model
   * @param[in] samples A distributed matrix containing samples for model input
   */
  template <typename DataT,
            El::Dist CDist,
            El::Dist RDist,
            El::DistWrap DistView,
            El::Device Device>
  void infer_mini_batch(
    model& model,
    El::DistMatrix<DataT, CDist, RDist, DistView, Device> const& samples)
  {
    for (int i = 0; i < model.get_num_layers(); i++) {
      auto& l = model.get_layer(i);
      // Insert samples into the input layer
      if (l.get_type() == "input") {
        auto& il = dynamic_cast<input_layer<DataType>&>(l);
        il.set_samples(samples);
      }
    }
    model.forward_prop(execution_mode::inference);
  }

  /** @brief Finds the predicted category in a models softmax layer
   * @param[in] model A model that has been used for inference
   * @param[in] labels A matrix to place predicted category labels
   */
  void get_labels(model& model, El::Matrix<int, El::Device::CPU>& labels)
  {
    int pred_label = 0;
    float max, col_value;

    for (const auto* l : model.get_layers()) {
      // Find the output layer
      if (l->get_type() == "softmax") {
        auto const& dtl =
          dynamic_cast<lbann::data_type_layer<float> const&>(*l);
        const auto& outputs = dtl.get_activations();

        // Find the prediction for each sample
        int col_count = outputs.Width();
        int row_count = outputs.Height();
        for (int i = 0; i < col_count; i++) {
          max = 0;
          for (int j = 0; j < row_count; j++) {
            col_value = outputs.Get(i, j);
            if (col_value > max) {
              max = col_value;
              pred_label = j;
            }
          }
          labels(i) = pred_label;
        }
      }
    }
  }
};

} // namespace lbann

#endif // LBANN_BATCH_INFERENCE_ALGORITHM_HPP
