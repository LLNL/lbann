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
#include "lbann/layers/lbann_layer_convolutional.hpp"

namespace lbann {

lbann_callback_imcomm::lbann_callback_imcomm(lbann_callback_imcomm::comm_type ct,
                                             lbann_summary* _summarizer) :
  lbann_callback(1, _summarizer), default_ct(ct) {
  set_name("imcomm");  
}

lbann_callback_imcomm::lbann_callback_imcomm(lbann_callback_imcomm::comm_type ct,
                                             std::unordered_set<uint> _layers,
                                             lbann_summary* _summarizer) :
  lbann_callback_imcomm(NONE, _summarizer) {
  for (const auto& layer : _layers) {
    param_choices[layer] = {};
    param_choices[layer].ct = ct;
  }
}

void lbann_callback_imcomm::set_layer_comm(uint layer, comm_type ct) {
  param_choices[layer] = {};
  param_choices[layer].ct = ct;
}

void lbann_callback_imcomm::set_layer_adaptive(uint layer, int proportion) {
  param_choices[layer] = {};
  param_choices[layer].ct = ADAPTIVE_QUANTIZATION;
  param_choices[layer].proportion = proportion;
}

void lbann_callback_imcomm::set_layer_threshold(
  uint layer, DataType pos_thresh, DataType neg_thresh) {
  param_choices[layer] = {};
  param_choices[layer].ct = THRESH_QUANTIZATION;
  param_choices[layer].pos_thresh = pos_thresh;
  param_choices[layer].neg_thresh = neg_thresh;
}

void lbann_callback_imcomm::setup(model* m) {
  std::vector<Layer*>& layers = m->get_layers();
  layer_params.resize(layers.size());
  for (size_t layer = 0; layer < layers.size(); ++layer) {
    imcomm_params& params = layer_params[layer];
    if (param_choices.find(layer) != param_choices.end()) {
      params = param_choices[layer];
    } else if (layer != 0 && layer != layers.size() - 1) {
      // Don't do communication for input/output layers unless explicitly told.
      // Also don't do communication for layers with no gradients.
      if (layers[layer]->get_weights_biases_gradient().Height() == 0) {
        params.ct = NONE;
      } else {
        params.ct = default_ct;
      }
    }
    if (params.ct != NONE) {
      // Update the effective mini-batch size so averaging is done properly.
      layers[layer]->set_effective_minibatch_size(
        layers[layer]->get_minibatch_size() * m->get_comm()->get_num_models());
      // Check if reshaping is needed.
      // Currently only automatically reshapes conv layers. (But ignores bias.)
      if (layers[layer]->m_type == layer_type::convolution) {
        convolutional_layer* conv_layer = (convolutional_layer*) layers[layer];
        params.reshape_height = conv_layer->m_num_input_channels *
          std::accumulate(conv_layer->m_filter_dims.begin(),
                          conv_layer->m_filter_dims.end(),
                          Int(0), std::multiplies<Int>());
        params.reshape_width = conv_layer->m_num_output_channels;
      }
      if (ct_does_quantization(params.ct)) {
        if (params.reshape_height) {
          Zeros(params.error, params.reshape_height, params.reshape_width);
        } else {
          const ElMat& gradients = layers[layer]->get_weights_biases_gradient();
          Zeros(params.error, gradients.LocalHeight(), gradients.LocalWidth());
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
  std::vector<Layer*>& layers = m->get_layers();
  for (size_t layer = 0; layer < layers.size(); ++layer) {
    imcomm_params& params = layer_params[layer];
    if (ct_does_quantization(params.ct)) {
      comm->intermodel_sum_matrix(params.error);
      Mat& local_gradients = layers[layer]->get_weights_biases_gradient().Matrix();
      if (params.reshape_height > 0) {
        Mat reshaped;
        reshape_mat(local_gradients, reshaped, params.reshape_height,
                    params.reshape_width);
        reshaped = params.error;
      } else {
        local_gradients = params.error;
      }
      // Apply optimizer update with accumulated gradient error.
      layers[layer]->update();
      Zero(params.error);
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
  for (size_t layer = 0; layer < layers.size(); ++layer) {
    double start_time = get_time();
    imcomm_params& params = layer_params[layer];
    if (params.ct == NONE) continue;
    Mat& local_gradients =
      layers[layer]->get_weights_biases_gradient().Matrix();
    Mat* reshaped = &local_gradients;
    if (params.reshape_height > 0 && ct_does_quantization(params.ct)) {
      if (layers[layer]->m_type == layer_type::convolution) {
        convolutional_layer* conv_layer = (convolutional_layer*) layers[layer];
        // Currently ignores the bias.
        Mat grad_view = local_gradients(IR(0, conv_layer->m_filter_size), ALL);
        reshape_mat(grad_view, *reshaped, params.reshape_height,
                    params.reshape_width);
      } else {
        reshape_mat(local_gradients, *reshaped, params.reshape_height,
                    params.reshape_width);
      }
    }
    switch (params.ct) {
    case NORMAL:
      comm->intermodel_sum_matrix(*reshaped);
      break;
    case ONEBIT_QUANTIZATION:
      quantizer.intermodel_sum_onebit_quantized(
        comm, *reshaped, params.error);
      break;
    case THRESH_QUANTIZATION:
      quantizer.intermodel_sum_threshold_quantized(
        comm, *reshaped, params.error, params.pos_thresh, params.neg_thresh);
      break;
    case ADAPTIVE_QUANTIZATION:
      quantizer.intermodel_sum_adaptive_quantized(
        comm, *reshaped, params.error, params.proportion);
      break;
    default:
      throw lbann_exception("imcomm: unknown comm type");
    }
    double im_time = get_time() - start_time;
    do_summary(m, layers[layer], im_time);
  }
}

void lbann_callback_imcomm::do_summary(model* m, Layer* layer,
                                       double im_time) {
  if (summarizer == nullptr) return;
  uint idx = layer->get_index();
  lbann_comm* comm = m->get_comm();
  std::string prefix = "layer" + std::to_string(
    static_cast<long long>(idx)) + "/imcomm_";
  summarizer->reduce_scalar(prefix + "time",
                            im_time, m->get_cur_step());
  size_t bytes_sent = 0;
  size_t bytes_received = 0;
  if (ct_does_quantization(layer_params[idx].ct)) {
    bytes_sent = comm->get_ar_bytes_sent();
    bytes_received = comm->get_ar_bytes_received();
  } else {
    // Use the same approximation the comm layer does.
    const Mat& local_gradients =
      layer->get_weights_biases_gradient().LockedMatrix();
    bytes_sent =
      sizeof(DataType) * local_gradients.Height() * local_gradients.Width();
    bytes_received =
      sizeof(DataType) * local_gradients.Height() * local_gradients.Width();
  }
  summarizer->reduce_scalar(prefix + "bytes_sent",
                            bytes_sent, m->get_cur_step());
  summarizer->reduce_scalar(prefix + "bytes_received",
                            bytes_received, m->get_cur_step());
  if (ct_does_quantization(layer_params[idx].ct)) {
    summarizer->reduce_scalar(prefix + "rs_bytes_sent",
                              comm->get_ar_rs_bytes_sent(),
                              m->get_cur_step());
    summarizer->reduce_scalar(prefix + "ag_bytes_sent",
                              comm->get_ar_ag_bytes_sent(),
                              m->get_cur_step());
    summarizer->reduce_scalar(prefix + "rs_bytes_received",
                              comm->get_ar_rs_bytes_received(),
                              m->get_cur_step());
    summarizer->reduce_scalar(prefix + "ag_bytes_received",
                              comm->get_ar_ag_bytes_received(),
                              m->get_cur_step());
    summarizer->reduce_scalar(prefix + "ar_send_trans_time",
                              comm->get_ar_send_transform_time(),
                              m->get_cur_step());
    summarizer->reduce_scalar(prefix + "ar_recv_trans_time",
                              comm->get_ar_recv_transform_time(),
                              m->get_cur_step());
    summarizer->reduce_scalar(prefix + "ar_recv_apply_trans_time",
                              comm->get_ar_recv_apply_transform_time(),
                              m->get_cur_step());
    if (layer_params[idx].ct == ADAPTIVE_QUANTIZATION) {
      summarizer->reduce_scalar(prefix + "quantized_count",
                                quantizer.get_quantized_count(),
                                m->get_cur_step());
    }
    quantizer.reset_counters();
    comm->reset_stats_counters();
  }
}

}  // namespace lbann
