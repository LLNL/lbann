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
// lbann_callback_summary .hpp .cpp - Callback hooks to summarize to Tensorboard
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/lbann_callback_summary.hpp"

namespace lbann {

lbann_callback_summary::lbann_callback_summary(lbann_summary* _summarizer,
                                               int _batch_interval) :
  lbann_callback(_batch_interval, _summarizer) {
  set_name("summary");  
}

void lbann_callback_summary::on_batch_end(model* m) {
  m->summarize(*summarizer);
  // Note that these comm stats are a running sum, so they count from the last
  // time we reset and thus are over the whole batch_interval period.
  // Bytes sent/received are the sum of the bytes sent/received by every rank
  // in a model.
  lbann_comm* comm = m->get_comm();
  size_t bytes_sent = comm->get_bytes_sent();
  size_t bytes_received = comm->get_bytes_received();
  size_t model_barriers = comm->get_num_model_barriers();
  size_t intermodel_barriers = comm->get_num_intermodel_barriers();
  size_t global_barriers = comm->get_num_global_barriers();
  comm->reset_stats_counters();
  summarizer->sum_reduce_scalar("bytes_sent", bytes_sent, m->get_cur_step());
  summarizer->sum_reduce_scalar("bytes_received", bytes_received,
                                m->get_cur_step());
  summarizer->reduce_scalar("model_barriers", model_barriers,
                            m->get_cur_step());
  summarizer->reduce_scalar("intermodel_barriers", intermodel_barriers,
                            m->get_cur_step());
  summarizer->reduce_scalar("global_barriers", global_barriers,
                            m->get_cur_step());
}

void lbann_callback_summary::on_epoch_end(model* m) {
  summarizer->reduce_scalar("train_accuracy", m->get_train_accuracy(),
                            m->get_cur_step());
  for (const auto& layer : m->get_layers()) {
    std::string prefix = "layer" + std::to_string(layer->get_index()) + "/";
    summarizer->reduce_histogram(prefix + "WB", layer->get_weights_biases(),
                                 m->get_cur_step());
  }
  summarizer->flush();
}

void lbann_callback_summary::on_test_end(model* m) {
  lbann_comm* comm = m->get_comm();
  summarizer->reduce_scalar("test_accuracy", m->get_test_accuracy(),
                            m->get_cur_step());
  // Reset counters incremented during test phase.
  comm->reset_stats_counters();
  for (auto&& layer : m->get_layers()) {
    layer->reset_counters();
  }
}

}  // namespace lbann
