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
//
// lbann_callback_imcomm .hpp .cpp - Send gradient updates between models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/lbann_callback_imcomm.hpp"
#include "lbann/utils/lbann_timer.hpp"
#include "lbann/utils/lbann_exception.hpp"

namespace lbann {

lbann_callback_imcomm::lbann_callback_imcomm(lbann_callback_imcomm::comm_type ct,
                                             lbann_summary* _summarizer) :
  lbann_callback(1, _summarizer), ct(ct) {
  
}

lbann_callback_imcomm::lbann_callback_imcomm(lbann_callback_imcomm::comm_type ct,
                                             std::unordered_set<uint> _layers,
                                             lbann_summary* _summarizer) :
  lbann_callback(1, _summarizer), ct(ct), layer_indices(_layers) {

}

void lbann_callback_imcomm::setup(Model* m) {
  if (ct != NONE) {
    bool add = layer_indices.size() == 0;
    std::vector<Layer*>& layers = m->get_layers();
    for (Layer* layer : layers) {
      uint idx = layer->get_index();
      if (add || layer_indices.find(idx) != layer_indices.end()) {
        // Ensure index is present (overwrites if already there).
        layer_indices.insert(idx);
        // Skip adding matrices when we don't need to.
        if (!ct_does_quantization()) continue;
        // TODO: handle case where WB_D is in other matrix distribution
        DistMat& WB_D = (DistMat&) layer->get_weights_biases_gradient();
        quantization_errors.emplace(idx, Mat{});
        Zeros(quantization_errors[idx], WB_D.LocalHeight(), WB_D.LocalWidth());
        im_quantization_errors.emplace(idx, Mat{});
        if (ct == ONEBIT_QUANTIZATION) {
          // Set up gradient history and SGD optimizer for one-bit quantization.
          gradhistories.emplace(idx, Mat{});
          if (layer->optimizer != nullptr) {
            if (typeid(*(layer->optimizer)) != typeid(Adagrad<DistMat>)) {
              throw lbann_exception(
                "lbann_callback_imcomm: Cannot do one-bit quantization for "
                "layer that does not use Adagrad");
            }
            // TODO: This leaks the old optimizer.
            layer->optimizer = new SGD<DistMat>(
              layer->comm, layer->optimizer->get_learning_rate(),
              0.0f, 0.0f, false);
            layer->optimizer->setup(layer->WB->Width(), layer->WB->Height());
          }
        }
      }
    }
  }
}

void lbann_callback_imcomm::on_epoch_end(Model* m) {
  lbann_comm* comm = m->get_comm();
  if (comm->get_num_models() == 1) {
    return;  // No point with only one model.
  }
  if (ct_does_quantization()) {
    std::vector<Layer*>& layers = m->get_layers();
    for (size_t l = 0; l < layers.size(); ++l) {
      if (layer_indices.find(layers[l]->get_index()) == layer_indices.end()) {
        continue;
      }
      comm->intermodel_sum_matrix(quantization_errors[l]);
      // TODO: handle case where WB_D is in other matrix distribution
      DistMat& WB_D = (DistMat&) layers[l]->get_weights_biases_gradient();
      Mat& local_mat = WB_D.Matrix();
      local_mat = quantization_errors[l];
      // Apply optimizer update again.
      layers[l]->update();
      Zeros(quantization_errors[l], quantization_errors[l].Height(),
            quantization_errors[l].Width());
    }
  }
}

void lbann_callback_imcomm::on_backward_prop_end(Model* m) {
  lbann_comm* comm = m->get_comm();
  if (comm->get_num_models() == 1) {
    return;  // No point with only one model.
  }
  std::vector<Layer*>& layers = m->get_layers();
  for (size_t l = 0; l < layers.size(); ++l) {
    if (layer_indices.find(layers[l]->get_index()) == layer_indices.end()) {
      continue;
    }
    double start_time = get_time();
    // TODO: handle case where WB_D is in other matrix distribution
    DistMat& WB_D = (DistMat&) layers[l]->get_weights_biases_gradient();
    switch (ct) {
    case NONE:
      break;
    case NORMAL:
      comm->intermodel_sum_matrix(WB_D);
      break;
    case ONEBIT_QUANTIZATION:
      quantizer.intermodel_sum_quantized(
        comm, WB_D, quantization_errors[l], im_quantization_errors[l], true,
        &(gradhistories[l]));
      break;
    case THRESH_QUANTIZATION:
      // TODO: Don't hardcode thresholds.
      quantizer.intermodel_sum_threshold_quantized(
        comm, WB_D, quantization_errors[l], 1.0f, -1.0f,
        im_quantization_errors[l], false);
      break;
    case COMPRESSED_THRESH_QUANTIZATION:
      // TODO: Don't hardcode thresholds.
      quantizer.intermodel_sum_threshold_quantized(
        comm, WB_D, quantization_errors[l], 1.0f, -1.0f,
        im_quantization_errors[l], true);
      break;
    case ADAPTIVE_THRESH_QUANTIZATION:
      // TODO: Don't hardcode proportion.
      quantizer.intermodel_sum_adaptive_threshold_quantized(
        comm, WB_D, quantization_errors[l], 5,
        im_quantization_errors[l], false);
      break;
    case COMPRESSED_ADAPTIVE_THRESH_QUANTIZATION:
      // TODO: Don't hardcode proportion.
      quantizer.intermodel_sum_adaptive_threshold_quantized(
        comm, WB_D, quantization_errors[l], 5,
        im_quantization_errors[l], true);
      break;
    }
    double im_time = get_time() - start_time;
    if (summarizer != nullptr && ct != NONE) {
      summarizer->reduce_scalar(
        "layer" + std::to_string(
          static_cast<long long>(layers[l]->get_index())) +
        "/imcomm_time",
        im_time, m->get_cur_step());
      summarizer->reduce_scalar(
        "layer" + std::to_string(
          static_cast<long long>(layers[l]->get_index())) +
        "/imcomm_bytes_sent",
        quantizer.get_bytes_sent(), m->get_cur_step());
      summarizer->reduce_scalar(
        "layer" + std::to_string(
          static_cast<long long>(layers[l]->get_index())) +
        "/imcomm_bytes_received",
        quantizer.get_bytes_received(), m->get_cur_step());
      quantizer.reset_bytes_counters();
    }
  }
}

}  // namespace lbann
