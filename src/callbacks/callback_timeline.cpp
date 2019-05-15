////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
// callback_timeline .hpp .cpp - Callback hooks to record a timeline of runtime
////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include "lbann/callbacks/callback_timeline.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

void lbann_callback_timeline::on_train_begin(model *m) {
  // Set up layers and weights.
  for (const auto& l : m->get_layers()) {
    m_fp_times.emplace(l->get_name(), std::vector<std::pair<EvalType,EvalType>>());
    m_bp_times.emplace(l->get_name(), std::vector<std::pair<EvalType,EvalType>>());
  }
  for (const auto& w : m->get_weights()) {
    m_opt_times.emplace(w->get_name(), std::vector<std::pair<EvalType,EvalType>>());
  }
  // Ensure the model is synchronized at the start.
  m->get_comm()->trainer_barrier();
  m_start_time = get_time();
}

void lbann_callback_timeline::on_train_end(model *m) {
  const std::string path = m_outdir + "/timeline.m" +
    std::to_string(m->get_comm()->get_trainer_rank()) + "." +
    std::to_string(m->get_comm()->get_rank_in_trainer()) + ".txt";
  std::ofstream f(path);
  for (const auto& kv : m_fp_times) {
    const std::string layer_name = "fp-" + kv.first;
    for (const auto& time : kv.second) {
      f << layer_name << ":" << time.first << ":" << time.second << '\n';
    }
  }
  for (const auto& kv : m_bp_times) {
    const std::string layer_name = "bp-" + kv.first;
    for (const auto& time : kv.second) {
      f << layer_name << ":" << time.first << ":" << time.second << '\n';
    }
  }
  for (const auto& kv : m_opt_times) {
    const std::string weights_name = "opt-" + kv.first;
    for (const auto& time : kv.second) {
      f << weights_name << ":" << time.first << ":" << time.second << '\n';
    }
  }
}

void lbann_callback_timeline::on_forward_prop_begin(model *m, Layer *l) {
  m_fp_start_time = get_rel_time();
}

void lbann_callback_timeline::on_forward_prop_end(model *m, Layer *l) {
  EvalType end = get_rel_time();
  m_fp_times[l->get_name()].emplace_back(m_fp_start_time, end);
}

void lbann_callback_timeline::on_backward_prop_begin(model *m, Layer *l) {
  m_bp_start_time = get_rel_time();
}

void lbann_callback_timeline::on_backward_prop_end(model *m, Layer *l) {
  EvalType end = get_rel_time();
  m_bp_times[l->get_name()].emplace_back(m_bp_start_time, end);
}

void lbann_callback_timeline::on_optimize_begin(model *m, weights *w) {
  m_opt_start_time = get_rel_time();
}

void lbann_callback_timeline::on_optimize_end(model *m, weights *w) {
  EvalType end = get_rel_time();
  m_opt_times[w->get_name()].emplace_back(m_opt_start_time, end);
}

}  // namespace lbann
