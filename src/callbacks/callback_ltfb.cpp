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

#include "lbann/callbacks/callback_ltfb.hpp"
#include "lbann/callbacks/callback_imcomm.hpp"
#include "lbann/metrics/categorical_accuracy.hpp"
#include "lbann/optimizers/adam.hpp"
#include "lbann/utils/random.hpp"
#include <typeinfo>
#include <typeindex>

namespace lbann {

namespace {

/** Assign partners for current tournament.
 *  This function pairs models up and returns the partner model
 *  corresponding to the current process. If there is an odd number of
 *  models, one of them is partnered with itself.
 */
int assign_partners(lbann_comm* comm) {
  if (comm->am_world_master()) {
    // Generate partner assignments on master process
    std::cout << "LTFB tournament partners:";
    const auto& num_models = comm->get_num_models();
    const auto& procs_per_model = comm->get_procs_per_model();
    std::vector<int> models(num_models);
    std::iota(models.begin(), models.end(), 0);
    std::shuffle(models.begin(), models.end(), get_fast_generator());
    std::vector<int> partners(num_models * procs_per_model);
    for (int i = 0; i < num_models; i += 2) {
      const auto& model1 = models[i];
      const auto& model2 = (i+1 < num_models) ? models[i+1] : model1;
      std::cout << (i > 0 ? "," : "") << " {" << model1;
      if (model1 != model2) { std::cout << "," << model2; }
      std::cout << "}";
      std::fill_n(partners.begin() + model1 * procs_per_model,
                  procs_per_model, model2);
      std::fill_n(partners.begin() + model2 * procs_per_model,
                  procs_per_model, model1);
    }
    std::cout << std::endl;
    return comm->scatter(partners.data(), comm->get_world_comm());
  } else {
    return comm->scatter<int>(0, comm->get_world_comm());
  }
}

/** Exchange weights with remote model.
 *  Weights from the local model are copied into local_weights and
 *  weights from the remote model are copied into model_weights.
 *  classic LTFB uses all weights, LTFB GAN uses selected weights
 */
void exchange_weights(lbann_comm* comm,
                      std::vector<weights*>& model_weights,
                      std::vector<weights*>& local_weights,
                      std::unordered_set<std::string>& selected_weights,
                      int partner) {
  for (size_t i = 0; i < model_weights.size(); ++i) {
    *local_weights[i] = *model_weights[i];
    const auto& local_matrix = local_weights[i]->get_values();
    auto&& remote_matrix = local_matrix.Copy();
    const auto& size = local_matrix.LocalHeight() * local_matrix.LocalWidth();
    bool send  = true;
    if(!selected_weights.empty()) {
      if(std::find(std::begin(selected_weights), std::end(selected_weights),
          model_weights[i]->get_name()) == std::end(selected_weights)) {
        send = false;
      }
    }
    if (size > 0 && send) {
      comm->sendrecv(local_matrix.LockedBuffer(), size, partner,
                     remote_matrix->Buffer(), size, partner);
      model_weights[i]->set_values(*remote_matrix);

      // Hack to communicate Adam state
      /// @todo Come up with something more general
      auto* local_opt = dynamic_cast<adam*>(local_weights[i]->get_optimizer());
      auto* remote_opt = dynamic_cast<adam*>(model_weights[i]->get_optimizer());
      if (local_opt != nullptr && remote_opt != nullptr) {
        comm->sendrecv(local_opt->m_moment1->LockedBuffer(), size, partner,
                       remote_opt->m_moment1->Buffer(), size, partner);
        comm->sendrecv(local_opt->m_moment2->LockedBuffer(), size, partner,
                       remote_opt->m_moment2->Buffer(), size, partner);
        std::vector<DataType> local_params(3), remote_params(3);
        local_params[0] = local_opt->get_learning_rate();
        local_params[1] = local_opt->m_beta1;
        local_params[2] = local_opt->m_beta2;
        comm->sendrecv(local_params.data(), 3, partner,
                       remote_params.data(), 3, partner);
        remote_opt->set_learning_rate(remote_params[0]);
        remote_opt->m_beta1 = remote_params[1];
        remote_opt->m_beta2 = remote_params[2];
      }
      
    }
    delete remote_matrix;
  }
}


/** Evaluate a model on tournament data and return evaluation metric value(s). */
/** @todo: deal with multiple metric values, return a list of values, max, min, mean? */
EvalType evaluate(model *m, std::unordered_set<std::string>& eval_metrics) {
  const auto& mode = m->get_execution_mode();
  m->evaluate(execution_mode::validation);
  m->set_execution_mode(mode);
  for (const auto& met : m->get_metrics()) {
    if(std::find(std::begin(eval_metrics), std::end(eval_metrics),
                  met->name()) != std::end(eval_metrics)) {
      return met->get_mean_value(execution_mode::validation);
    }
  }
  return EvalType(0);
}

} // namespace

lbann_callback_ltfb::lbann_callback_ltfb(int round_size, 
                                         std::unordered_set<std::string> eval_metrics,
                                         bool increasing_metric_mode,
                                         std::unordered_set<std::string> weights_tosend, 
                                         lbann_summary *summarizer)
  : lbann_callback(1, summarizer), m_round_size(round_size), 
                   m_eval_metrics(std::move(eval_metrics)),
                   m_increasing_metric_mode(increasing_metric_mode),
                   m_weights_tosend(std::move(weights_tosend)){}

lbann_callback_ltfb::lbann_callback_ltfb(const lbann_callback_ltfb& other) :
  lbann_callback(other),
  m_comm(other.m_comm),
  m_round_size(other.m_round_size),
  m_eval_metrics(other.m_eval_metrics),
  m_increasing_metric_mode(other.m_increasing_metric_mode),
  m_weights_tosend(other.m_weights_tosend),
  m_local_weights(other.m_local_weights) {
  for (auto& w : m_local_weights) { w = w->copy(); }
}

lbann_callback_ltfb& lbann_callback_ltfb::operator=(const lbann_callback_ltfb& other) {

  // Shallow copies
  m_comm = other.m_comm;
  m_round_size = other.m_round_size;
  m_eval_metrics = other.m_eval_metrics;
  m_increasing_metric_mode = other.m_increasing_metric_mode; 
  m_weights_tosend = other.m_weights_tosend;

  // Deep copy
  for (auto& w : m_local_weights) { delete w; }
  m_local_weights = other.m_local_weights;
  for (auto& w : m_local_weights) { w = w->copy(); }

  return *this;
}

lbann_callback_ltfb::~lbann_callback_ltfb() {
  for (auto& w : m_local_weights) { delete w; }
}

void lbann_callback_ltfb::setup(model *m) {
  
  if(m_eval_metrics.size() < 1)
    LBANN_ERROR("LTFB: specify at least one evaluation metric for tournament voting."); 

  m_comm = m->get_comm();

  // Create copy of model weights
  /// @todo Support LTFB with different models
  for (auto& w : m_local_weights) { delete w; }
  m_local_weights = m->get_weights();
  for (auto& w : m_local_weights) { w = w->copy(); }
  
  // Make sure model does not have inter-model communication callback
  for (auto&& cb : m->get_callbacks()) {
    if (dynamic_cast<lbann_callback_imcomm*>(cb) != nullptr) {
      LBANN_ERROR("Detected both LTFB and imcomm callbacks. ");
    }
  }

}

void lbann_callback_ltfb::on_batch_begin(model *m) {

  // Check whether to start LTFB round
  const auto& mode = m->get_execution_mode();
  const auto& step = m->get_cur_step();
  if (mode != execution_mode::training) { return; }
  if (step % m_round_size != 0 || step == 0) { return; }
  if (m_comm->am_world_master()) {
    std::cout << "---- LTFB round (step " << step << ") ----" << std::endl;
  }

  // Determine partner model for tournament
  const auto& local_model = m_comm->get_model_rank();
  const auto& remote_model = assign_partners(m_comm);
  if (remote_model == local_model) { return; }

  // Evaluate local model on tournament data
  if (m_comm->am_world_master()) {
    std::cout << "LTFB: evaluating local model..." << std::endl;
  }
  /** @todo: deal with multiple metric values, return a list of values, max, min, mean? */
  const auto& local_score = evaluate(m, m_eval_metrics);
  // Evaluate remote model on tournament data
  if (m_comm->am_world_master()) {
    std::cout << "LTFB: evaluating remote model..." << std::endl;
  }
  auto model_weights = m->get_weights();
  // Note: (Selected/all) weights from remote model are copied into local model
  //GAN: only send weights specified in prototext (e.g., generator or discriminator)
  exchange_weights(m_comm, model_weights, m_local_weights,m_weights_tosend, remote_model);

  //evaluate received model on tournament data
  const auto& remote_score = evaluate(m, m_eval_metrics);

  // Restore local weights if they achieve a better score
  if((m_increasing_metric_mode && remote_score <= local_score) ||
      (!m_increasing_metric_mode  && remote_score >= local_score)) {
    for (size_t i = 0; i < model_weights.size(); ++i) {
      *model_weights[i] = *m_local_weights[i];
    }
  } else {
    if (m_comm->am_model_master()) {
      std::cout << "LTFB: replacing model " << local_model << " "
                << "(" << local_score << " score) "
                << "with model " << remote_model << " "
                << "(" << remote_score << " score) "
                << std::endl;
    }
  }

}

}  // namespace lbann
