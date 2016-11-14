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

void lbann_callback_imcomm::setup(model* m) {
  if (ct != NONE) {
    bool add = layer_indices.size() == 0;
    std::vector<Layer*>& layers = m->get_layers();
    for (Layer* layer : layers) {
      uint idx = layer->get_index();
      if (add || layer_indices.find(idx) != layer_indices.end()) {
        // Ensure index is present (overwrites if already there).
        layer_indices.insert(idx);
        // Update the layer's effective mini-batch size so it averages properly.
        layer->set_effective_minibatch_size(
          layer->get_minibatch_size() * m->get_comm()->get_num_models());
        // Skip adding matrices when we don't need to.
        if (!ct_does_quantization()) continue;
        // TODO: handle case where weights_gradient is in other matrix distribution
        DistMat& weights_gradient = (DistMat&) layer->get_weights_biases_gradient();
        quantization_errors.emplace(idx, Mat{});
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
            layer->optimizer->setup(layer->m_weights->Width(), layer->m_weights->Height());
          }
        }
      }
    }
  }
}

void lbann_callback_imcomm::on_epoch_end(model* m) {
  lbann_comm* comm = m->get_comm();
  if (comm->get_num_models() == 1 ||
      m->get_execution_mode() != execution_mode::training) {
    return;  // No point with only one model.
  }
  if (ct_does_quantization()) {
    std::vector<Layer*>& layers = m->get_layers();
    for (size_t l = 0; l < layers.size(); ++l) {
      if (layer_indices.find(layers[l]->get_index()) == layer_indices.end()) {
        continue;
      }
      comm->intermodel_sum_matrix(quantization_errors[l]);
      // TODO: handle case where weights_gradient is in other matrix distribution
      DistMat& weights_gradient = (DistMat&) layers[l]->get_weights_biases_gradient();
      Mat& local_mat = weights_gradient.Matrix();
      local_mat = quantization_errors[l];
      // Apply optimizer update again.
      layers[l]->update();
      quantization_errors[l].Empty();
    }
  }
}

void lbann_callback_imcomm::on_backward_prop_end(model* m) {
  lbann_comm* comm = m->get_comm();
  if (comm->get_num_models() == 1 ||
      m->get_execution_mode() != execution_mode::training) {
    return;  // No point with only one model.
  }
  std::vector<Layer*>& layers = m->get_layers();
  for (size_t l = 0; l < layers.size(); ++l) {
    if (layer_indices.find(layers[l]->get_index()) == layer_indices.end()) {
      continue;
    }
    double start_time = get_time();
    // TODO: handle case where weights_gradient is in other matrix distribution
    DistMat& weights_gradient = (DistMat&) layers[l]->get_weights_biases_gradient();
    switch (ct) {
    case NONE:
      break;
    case NORMAL:
      comm->intermodel_sum_matrix(weights_gradient);
      break;
    case ONEBIT_QUANTIZATION:
      quantizer.intermodel_sum_quantized(
        comm, weights_gradient, quantization_errors[l], im_quantization_errors[l], true,
        &(gradhistories[l]));
      break;
    case THRESH_QUANTIZATION:
      // TODO: Don't hardcode thresholds.
      quantizer.intermodel_sum_threshold_quantized(
        comm, weights_gradient, quantization_errors[l], 0.01f, -0.01f,
        im_quantization_errors[l], false);
      break;
    case COMPRESSED_THRESH_QUANTIZATION:
      // TODO: Don't hardcode thresholds.
      quantizer.intermodel_sum_threshold_quantized(
        comm, weights_gradient, quantization_errors[l], 0.01f, -0.01f,
        im_quantization_errors[l], true);
      break;
    case ADAPTIVE_THRESH_QUANTIZATION:
      // TODO: Don't hardcode proportion.
      quantizer.intermodel_sum_adaptive_threshold_quantized(
        comm, weights_gradient, quantization_errors[l], 64,
        im_quantization_errors[l], false);
      break;
    case COMPRESSED_ADAPTIVE_THRESH_QUANTIZATION:
      // TODO: Don't hardcode proportion.
      quantizer.intermodel_sum_adaptive_threshold_quantized(
        comm, weights_gradient, quantization_errors[l], 64,
        im_quantization_errors[l], true);
      break;
    }
    double im_time = get_time() - start_time;
    if (summarizer != nullptr && ct != NONE) {
      std::string prefix = "layer" + std::to_string(
        static_cast<long long>(layers[l]->get_index())) + "/imcomm_";
      summarizer->reduce_scalar(prefix + "time",
                                im_time, m->get_cur_step());
      size_t bytes_sent = 0;
      size_t bytes_received = 0;
      if (ct_does_quantization()) {
        bytes_sent = quantizer.get_bytes_sent();
        bytes_received = quantizer.get_bytes_received();
      } else {
        // Use the same approximation the comm layer does.
        bytes_sent = sizeof(DataType) * weights_gradient.LocalHeight() * weights_gradient.LocalWidth();
        bytes_received = sizeof(DataType) * weights_gradient.LocalHeight() * weights_gradient.LocalWidth();
      }
      summarizer->reduce_scalar(prefix + "bytes_sent",
                                bytes_sent, m->get_cur_step());
      summarizer->reduce_scalar(prefix + "bytes_received",
                                bytes_received, m->get_cur_step());
      if (ct_does_quantization()) {
        summarizer->reduce_scalar(prefix + "rs_bytes_sent",
                                  quantizer.get_rs_bytes_sent(),
                                  m->get_cur_step());
        summarizer->reduce_scalar(prefix + "ag_bytes_sent",
                                  quantizer.get_ag_bytes_sent(),
                                  m->get_cur_step());
        summarizer->reduce_scalar(prefix + "rs_bytes_received",
                                  quantizer.get_rs_bytes_received(),
                                  m->get_cur_step());
        summarizer->reduce_scalar(prefix + "ag_bytes_received",
                                  quantizer.get_ag_bytes_received(),
                                  m->get_cur_step());
        summarizer->reduce_scalar(prefix + "rs_send_trans_time",
                                  quantizer.get_rs_send_trans_time(),
                                  m->get_cur_step());
        summarizer->reduce_scalar(prefix + "rs_recv_trans_time",
                                  quantizer.get_rs_recv_trans_time(),
                                  m->get_cur_step());
        summarizer->reduce_scalar(prefix + "ag_recv_trans_time",
                                  quantizer.get_ag_recv_trans_time(),
                                  m->get_cur_step());
        quantizer.reset_bytes_counters();
        quantizer.reset_time_counters();
      }
    }
  }
}

}  // namespace lbann
