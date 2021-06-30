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

#include "lbann/execution_algorithms/kfac.hpp"
#include "lbann/execution_algorithms/kfac/execution_context.hpp"
#include "lbann/execution_algorithms/sgd_training_algorithm.hpp"

#include "lbann/base.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/execution_algorithms/kfac/kfac_block.hpp"
#include "lbann/execution_algorithms/kfac/kfac_block_bn.hpp"
#include "lbann/execution_algorithms/kfac/kfac_block_fc_conv.hpp"
#include "lbann/execution_algorithms/kfac/kfac_block_gru.hpp"
#include "lbann/execution_algorithms/kfac/kfac_util.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/profiling.hpp"

#include <training_algorithm.pb.h>

#include <cstddef>
#include <limits>

namespace lbann {

/// @todo Initialize properly
KFAC::KFAC(
  std::string name,
  std::unique_ptr<TermCriteriaType> stop,
  std::vector<double> damping_act_params,
  std::vector<double> damping_err_params,
  std::vector<double> damping_bn_act_params,
  std::vector<double> damping_bn_err_params,
  size_t damping_warmup_steps,
  double kronecker_decay,
  bool print_time,  bool print_matrix, bool print_matrix_summary,
  bool use_pi,
  std::vector<size_t> update_intervals,
  size_t update_interval_steps,
  kfac::kfac_inverse_strategy inverse_strategy,
  std::vector<std::string> disable_layers,
  double learning_rate_factor,
  double learning_rate_factor_gru)
  : BaseType{std::move(name)},
    m_stopping_criteria{std::move(stop)},
    m_damping_act_params{std::move(damping_act_params)},
    m_damping_err_params{std::move(damping_err_params)},
    m_damping_bn_act_params{std::move(damping_bn_act_params)},
    m_damping_bn_err_params{std::move(damping_bn_err_params)},
    m_damping_warmup_steps{std::move(damping_warmup_steps)},
    m_kronecker_decay{kronecker_decay},
    m_print_time{print_time},
    m_print_matrix{print_matrix},
    m_print_matrix_summary{print_matrix_summary},
    m_use_pi{use_pi},
    m_update_intervals{std::move(update_intervals)},
    m_update_interval_steps{update_interval_steps},
    m_inverse_strategy{inverse_strategy},
    m_disable_layers{std::move(disable_layers)},
    m_learning_rate_factor{learning_rate_factor},
    m_learning_rate_factor_gru{learning_rate_factor_gru}
{}

KFAC::KFAC(KFAC const& other)
  : BaseType(other.get_name()),
    m_stopping_criteria{other.m_stopping_criteria->clone()},
    m_damping_act_params{other.m_damping_act_params},
    m_damping_err_params{other.m_damping_err_params},
    m_damping_bn_act_params{other.m_damping_bn_act_params},
    m_damping_bn_err_params{other.m_damping_bn_err_params},
    m_damping_warmup_steps{other.m_damping_warmup_steps},
    m_kronecker_decay{other.m_kronecker_decay},
    m_print_time{other.m_print_time},
    m_print_matrix{other.m_print_matrix},
    m_print_matrix_summary{other.m_print_matrix_summary},
    m_use_pi{other.m_use_pi},
    m_update_intervals{other.m_update_intervals},
    m_update_interval_steps{other.m_update_interval_steps},
    m_inverse_strategy{other.m_inverse_strategy},
    m_disable_layers{other.m_disable_layers},
    m_learning_rate_factor{other.m_learning_rate_factor}
{}

KFAC& KFAC::operator=(KFAC const& other) {
  BaseType::operator=(other);
  m_stopping_criteria = other.m_stopping_criteria->clone();
  m_damping_act_params = other.m_damping_act_params;
  m_damping_err_params = other.m_damping_err_params;
  m_damping_bn_act_params = other.m_damping_bn_act_params;
  m_damping_bn_err_params = other.m_damping_bn_err_params;
  m_damping_warmup_steps = other.m_damping_warmup_steps;
  m_kronecker_decay = other.m_kronecker_decay;
  m_print_time = other.m_print_time;
  m_print_matrix = other.m_print_matrix;
  m_print_matrix_summary = other.m_print_matrix_summary;
  m_use_pi = other.m_use_pi;
  m_update_intervals = other.m_update_intervals;
  m_update_interval_steps = other.m_update_interval_steps;
  m_inverse_strategy = other.m_inverse_strategy;
  m_disable_layers = other.m_disable_layers;
  m_learning_rate_factor = other.m_learning_rate_factor;
  return *this;
}

std::string KFAC::get_type() const { return "KFAC"; }

kfac::ExecutionContext* KFAC::do_get_new_execution_context() const
{
  return new kfac::ExecutionContext(
    0UL,
    m_damping_act_params[0],
    m_damping_err_params[0],
    m_damping_bn_act_params[0],
    m_damping_bn_err_params[0]);
}

// =============================================
// Evaluation and training
// =============================================

void KFAC::apply(
  execution_context& context_,
  model& model,
  data_coordinator& dc,
  execution_mode mode)
{
  ExeContextType& context = dynamic_cast<ExeContextType&>(context_);
  if (mode == execution_mode::training) {
    train(context, model, dc, *m_stopping_criteria);
  }
  else {
    sgd_training_algorithm eval_algo(
      this->get_name()+"_eval",
      m_stopping_criteria->clone());
    auto& eval_context = context.get_sgd_execution_context();
    eval_algo.apply(eval_context, model, dc, mode);
  }
}

void KFAC::train(
  ExeContextType& kfac_context,
  model& model,
  data_coordinator& dc,
  TermCriteriaType const& term)
{
  // Initialize some state so it knows we're training now.
  auto& sgd_context = kfac_context.get_sgd_execution_context();
  sgd_context.set_execution_mode(execution_mode::training);
  model.reset_mode(sgd_context, execution_mode::training);
  dc.reset_mode(sgd_context);

  // Reset KFAC context
  kfac_context.m_damping_act = m_damping_act_params[0];
  kfac_context.m_damping_err = m_damping_err_params[0];
  kfac_context.m_damping_bn_act = m_damping_bn_act_params[0];
  kfac_context.m_damping_bn_err = m_damping_bn_err_params[0];

  // Run callbacks.
  do_train_begin_cbs(model);

  // Start iterating
  bool is_start_of_epoch = true;
  sgd_context.start_timer();
  while (!term(sgd_context)) {

    if (is_start_of_epoch) {
      // Initialize epoch
      model.reset_mode(sgd_context, execution_mode::training);
      model.reset_epoch_statistics(execution_mode::training);
      dc.reset_mode(sgd_context);
      do_epoch_begin_cbs(model);
      is_start_of_epoch = false;
    }

    // Train a mini batch. Returns "true" if the data_coordinator
    // detects the end of an epoch.
    if (train_mini_batch(kfac_context, model, dc)) {
      // Finalize epoch
      sgd_context.inc_epoch();
      model.reconcile_weight_values();
      do_epoch_end_cbs(model);

      // Evaluate on validation set
      //
      // FIXME (trb 05/04/2021): Upon further refactor, this should
      // move out of the main training cycle and become part of an
      // "evaluation policy" or something of that nature, ideally with
      // its own context that we needn't know about.
      if (dc.is_execution_mode_valid(execution_mode::validation)) {
        const execution_mode eval_mode = execution_mode::validation;
        sgd_execution_context eval_context(
          eval_mode,
          dc.get_mini_batch_size(eval_mode));
        // FIXME (trb 05/05/2021): This hacks around a bad assumption
        // in the data store.
        // Note (tym 6/7/21): Copied from sgd_training_algorithm.cpp.
        size_t num_validation_epochs = 1UL;
        if (sgd_context.get_epoch() > 1UL) {
          eval_context.inc_epoch();
          ++num_validation_epochs;
        }
        sgd_training_algorithm eval_algo(
          this->get_name()+"_eval",
          make_unique<epoch_termination_criteria>(num_validation_epochs));
        eval_algo.apply(eval_context, model, dc, eval_mode);

        // FIXME (trb 06/07/21): The early stopping callback is part
        // of the evaluation callbacks but it's meant to affect
        // training. This fixes a bug in which the training context
        // was meant to stop but was never properly told.
        sgd_context.set_early_stop(eval_context.get_early_stop());

      }

      // Trigger new epoch stuff next iteration (if there is one).
      is_start_of_epoch = true;
    }
  }
  sgd_context.stop_timer();

  // Reset the model back to the training execution context prior to
  // end of training callbacks
  model.reset_mode(sgd_context, execution_mode::training);
  do_train_end_cbs(model);
}

// =============================================
// Mini-batch step
// =============================================

// Returns "true" if the data_coordinator detects the end of an epoch.
bool KFAC::train_mini_batch(
  ExeContextType& kfac_context,
  model& model,
  data_coordinator& dc)
{
  auto& sgd_context = kfac_context.get_sgd_execution_context();

  model.reset_mode(sgd_context, execution_mode::training);
  dc.reset_mode(sgd_context);
  do_batch_begin_cbs(model);

  bool finished = false;

  dc.fetch_data(execution_mode::training);

#if defined(LBANN_HAVE_OMP_TASKLOOP)
  LBANN_OMP_PARALLEL
  {
#pragma omp single
    {
#endif
      // Forward prop step
      model.clear_gradients();
      model.forward_prop(execution_mode::training);
      on_forward_prop_end(kfac_context, model);
      // check if the data coordinator has finished the epoch and kickoff
      // background I/O
      finished = dc.epoch_complete(execution_mode::training);

      // Result is not needed until the end of the mini-batch.
      model.get_objective_function()->start_evaluation(
        execution_mode::training,
        sgd_context.get_current_mini_batch_size());

      // Backward prop step
      model.get_objective_function()->differentiate();
      model.backward_prop();
      on_backward_prop_end(kfac_context, model);
      model.get_objective_function()->compute_weight_regularization();

      // Finish evaluation.
      model.get_objective_function()->finish_evaluation(
        execution_mode::training,
        sgd_context.get_current_mini_batch_size());
      model.evaluate_metrics(execution_mode::training,
                             sgd_context.get_current_mini_batch_size());

      // Update step
      model.update_weights();
      model.update_layers();
#if defined(LBANN_HAVE_OMP_TASKLOOP)
    }
  }
#endif

  kfac_context.inc_step();
  sgd_context.inc_step();
  do_batch_end_cbs(model);
  return finished;
}

// =============================================
// Callbacks
// =============================================

void KFAC::do_train_begin_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_begin(&model);
  }
}

void KFAC::do_train_end_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_end(&model);
  }
}

void KFAC::do_epoch_begin_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_begin(&model);
  }
}

void KFAC::do_epoch_end_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_end(&model);
  }
}

void KFAC::do_batch_begin_cbs(model& model)
{
  sgd_execution_context& c =
    static_cast<sgd_execution_context&>(model.get_execution_context());
  for (const auto& cb : model.get_callbacks()) {
    if (c.get_step() % cb->get_batch_interval() == 0) {
      cb->on_batch_begin(&model);
    }
  }
}

void KFAC::do_batch_end_cbs(model& model)
{
  sgd_execution_context& c =
    static_cast<sgd_execution_context&>(model.get_execution_context());
  for (const auto& cb : model.get_callbacks()) {
    if (c.get_step() % cb->get_batch_interval() == 0) {
      cb->on_batch_end(&model);
    }
  }
}

// =============================================
// KFAC implementation
// =============================================

void KFAC::on_forward_prop_end(
  ExeContextType& context,
  model& model) {

  auto& comm = *model.get_comm();
  const auto layers = model.get_layers();

  // List up layers to be updated
  if(context.m_blocks.size() == 0){
    prof_region_begin("kfac-setup", prof_color, prof_sync);
    const size_t num_procs = comm.get_procs_per_trainer();
    std::unordered_map<std::string, int> proc_ranks;
    for(auto i_layer = layers.begin(); i_layer != layers.end(); i_layer++) {
      const size_t layer_id = std::distance(layers.begin(), i_layer);
      const auto &l = *i_layer;
      const auto l_fc = dynamic_cast<fully_connected_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(l);
      const auto l_conv = dynamic_cast<convolution_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(l);
      const auto l_bn = dynamic_cast<batch_normalization_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(l);
      const auto l_gru = dynamic_cast<gru_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(l);
      const bool is_fc = (l_fc != nullptr);
      const bool is_conv = (l_conv != nullptr);
      const bool is_bn = (l_bn != nullptr);
      const bool is_gru = (l_gru != nullptr);
      if(!(is_fc || is_conv || is_bn || is_gru))
        continue;

      if(std::find(m_disable_layers.begin(), m_disable_layers.end(), l->get_name()) != m_disable_layers.end()) {
        if(comm.am_trainer_master())
          std::cout << "K-fac: " << l->get_name() << " is ignored to optimize with K-FAC." << std::endl;
        continue;
      }

      prof_region_begin(("kfac-setup/" + l->get_name()).c_str(), prof_color, prof_sync);

      // Ignore layers without optimizers
      const auto& weights = l->get_weights(0);
      const optimizer *w_optimizer = weights.get_optimizer();
      if(w_optimizer == nullptr)
        continue;

      std::string proc_rank_key = "all";
      if(m_inverse_strategy == kfac::kfac_inverse_strategy::EACH)
        proc_rank_key = l->get_type();
      if(proc_ranks.find(proc_rank_key) == proc_ranks.end())
        proc_ranks[proc_rank_key] = 0;
      int& proc_rank = proc_ranks[proc_rank_key];

      // Check layer property
      if((l->get_num_parents() != 1 || l->get_num_children() != 1) && !is_gru) {
        std::stringstream err;
        err << "K-FAC expects layers who have exact one parent and child."
            << " layer: " << l->get_name()
            << ", #parent: " << l->get_num_parents()
            << ", #child: " << l->get_num_children();
        LBANN_ERROR(err.str());
      }

      std::shared_ptr<kfac_block<Device>> block;
      if(is_fc || is_conv) {
        block = std::make_shared<kfac_block_fc_conv<Device>>(
            l, &context, layer_id, proc_rank, is_conv);
      } else if(is_bn) {
        block = std::make_shared<kfac_block_bn<Device>>(
            l, &context, layer_id, proc_rank);
      } else if(is_gru) {
        block = std::make_shared<kfac_block_gru<Device>>(
            l, &context, layer_id, proc_rank);
      }

      context.m_blocks.push_back(std::move(block));
      if(m_inverse_strategy != kfac::kfac_inverse_strategy::ROOT)
        proc_rank = (proc_rank+1)%num_procs;

      prof_region_end(("kfac-setup/" + l->get_name()).c_str(), prof_sync);
    }

    if(comm.am_trainer_master()) {
      for(const auto& block : context.m_blocks)
        std::cout << "K-FAC setup: "
                  << block->get_info() << std::endl;
    }

    prof_region_end("kfac-setup", prof_sync);
  }

  for(auto& block : context.m_blocks)
    block->on_forward_prop_end();

}

void KFAC::on_backward_prop_end(
  ExeContextType& context,
  model& model) {

  // Get some configs
  auto& comm = *model.get_comm();
  const auto& sgd_context = context.get_sgd_execution_context();
  const size_t num_steps = sgd_context.get_step();
  const auto layers = model.get_layers();

  // Update the damping value
  // using a modified Tikhonov damping tequnique from
  // http://arxiv.org/abs/1811.12019
  const auto get_next_damping
    = [](const double damping_prev,
         const std::vector<double> damping_params,
         const double damping_warmup_steps) {
    if(damping_params.size() == 1)
      return damping_params[0];
    const DataType alpha = 2.0 * log10(damping_params[0] / damping_params[1]) / damping_warmup_steps;
    return (1.0-alpha) * damping_prev + alpha * damping_params[1];
  };
  context.m_damping_act = get_next_damping(
    context.m_damping_act, m_damping_act_params, m_damping_warmup_steps);
  context.m_damping_err = get_next_damping(
    context.m_damping_err, m_damping_err_params, m_damping_warmup_steps);
  context.m_damping_bn_act = get_next_damping(
    context.m_damping_bn_act, m_damping_bn_act_params, m_damping_warmup_steps);
  context.m_damping_bn_err = get_next_damping(
    context.m_damping_bn_err, m_damping_bn_err_params, m_damping_warmup_steps);

  // Update the udpate interval
  if(m_update_intervals.size() == 1)
    context.m_update_interval = m_update_intervals[0];
  else {
    context.m_update_interval = m_update_intervals[0]
        + ((double) m_update_intervals[1]-m_update_intervals[0])
        * std::min((double) num_steps/ m_update_interval_steps, 1.0);
  }

  // List up layers to be updated
  if(context.m_blocks.size() == 0){
    LBANN_ERROR("K-FAC blocks have not been setup");
  }

  prof_region_begin("kfac-step", prof_color, prof_sync);

  // Step 1: Ensure that each process has averaged Kronecker factors
  // for the model-parallel part.
  const bool is_first_step = (!m_has_kronecker_inverse);
  const bool is_kronecker_update_required =
      ((num_steps%context.m_update_interval) == 0 || !m_has_kronecker_inverse);
  if(is_kronecker_update_required) {
    prof_region_begin("kfac-update", prof_color, prof_sync);

    prof_region_begin("kfac-update/local", prof_color, prof_sync);
    for(auto& block : context.m_blocks) {
      prof_region_begin(("kfac-update/local/" + block->get_name()).c_str(), prof_color, prof_sync);
      block->compute_local_kronecker_factors(
        &comm, m_print_matrix, m_print_matrix_summary);
      prof_region_end(("kfac-update/local/" + block->get_name()).c_str(), prof_sync);
    }
    prof_region_end("kfac-update/local", prof_sync);

#ifdef LBANN_NVPROF
    prof_region_begin("kfac-update/local-barrier", prof_color, prof_sync);
    CHECK_CUDA(cudaDeviceSynchronize());
    comm.trainer_barrier();
    prof_region_end("kfac-update/local-barrier", prof_sync);
#endif // LBANN_NVPROF

    // List-up buffers to synchronize.
    std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> buffers;
    size_t global_buffer_size = 0;
    for(auto& block : context.m_blocks)
      for(auto L : block->get_local_kronecker_buffers()) {
        const size_t rank = block->get_inverse_proc_rank();
        buffers.emplace_back(rank, L);
        assert(L->Width() == 1);
        global_buffer_size += L->Height();
      }

    // Perform reduce-scatter.
    prof_region_begin("kfac-update/reduce-scatter", prof_color, prof_sync);
    const auto reduce_scatter_mode = kfac::kfac_reduce_scatter_mode::ALLREDUCE;
    El::Matrix<DataType, Device>& global_buffer =
      context.get_workspace_matrix(
        "reduce_scatter_send_buffer",
        kfac::is_reduce_scatter_buffer_required(reduce_scatter_mode) ? global_buffer_size : 0,
        1);
    kfac::reduce_scatter_blocks(
        buffers, global_buffer, &comm, reduce_scatter_mode);
    prof_region_end("kfac-update/reduce-scatter", prof_sync);

#ifdef LBANN_NVPROF
    prof_region_begin("kfac-update/reduce-scatter-barrier", prof_color, prof_sync);
    CHECK_CUDA(cudaDeviceSynchronize());
    comm.trainer_barrier();
    prof_region_end("kfac-update/reduce-scatter-barrier", prof_sync);
#endif // LBANN_NVPROF

    prof_region_begin("kfac-update/average", prof_color, prof_sync);
    for(auto& block : context.m_blocks) {
      prof_region_begin(("kfac-update/average/" + block->get_name()).c_str(), prof_color, prof_sync);
      block->update_kronecker_average(
          &comm,
          m_kronecker_decay,
          m_print_matrix, m_print_matrix_summary);
      prof_region_end(("kfac-update/average/" + block->get_name()).c_str(), prof_sync);
    }
    prof_region_end("kfac-update/average", prof_sync);

    prof_region_end("kfac-update", prof_sync);
  }

  // Step 2: Model-parallel inverse computation
  prof_region_begin("kfac-inverse", prof_color, prof_sync);
  for(auto& block : context.m_blocks) {
    if(!is_kronecker_update_required || (size_t) comm.get_rank_in_trainer() != block->get_inverse_proc_rank())
      continue;

    prof_region_begin(("kfac-inverse/" + block->get_name()).c_str(), prof_color, prof_sync);
    // TODO: Add kfac_block::is_bn?
    const bool is_bn = dynamic_cast<kfac_block_bn<Device>*>(block.get()) != nullptr;
    const bool is_gru = dynamic_cast<kfac_block_gru<Device>*>(block.get()) != nullptr;
    block->update_kronecker_inverse(
        &comm, m_use_pi,
        is_bn ? context.m_damping_bn_act : context.m_damping_act,
        is_bn ? context.m_damping_bn_err : context.m_damping_err,
        is_gru ? m_learning_rate_factor_gru : m_learning_rate_factor,
        m_print_matrix, m_print_matrix_summary,
        m_print_time);
    prof_region_end(("kfac-inverse/" + block->get_name()).c_str(), prof_sync);
  }
  m_has_kronecker_inverse = true;
  prof_region_end("kfac-inverse", prof_sync);

#ifdef LBANN_NVPROF
  prof_region_begin("kfac-inverse-barrier", prof_color, prof_sync);
  CHECK_CUDA(cudaDeviceSynchronize());
  comm.trainer_barrier();
  prof_region_end("kfac-inverse-barrier", prof_sync);
#endif // LBANN_NVPROF

  // Step 3: All-gather of each preconditioned gradient tensor
  prof_region_begin("kfac-allgather", prof_color, prof_sync);
  {
    // List-up buffers to synchronize.
    std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> buffers;
    int local_buffer_size = 0, global_buffer_size = 0;
    for(auto& block : context.m_blocks)
      for(auto L : block->get_preconditioned_grad_buffers()) {
        const size_t rank = block->get_inverse_proc_rank();
        buffers.emplace_back(rank, L);
        assert(L->Width() == 1);
        if(rank == (size_t) comm.get_rank_in_trainer())
          local_buffer_size += L->Height();
        global_buffer_size += L->Height();
      }

    // Perform allgather.
    const auto allgather_mode = kfac::kfac_allgather_mode::ALLREDUCE;
    const auto is_buffer_needed = kfac::is_allgather_buffer_required(allgather_mode);
    El::Matrix<DataType, Device>& local_buffer =
      context.get_workspace_matrix(
        "allgather_send_buffer",
        is_buffer_needed.first ? local_buffer_size : 0,
        1);
    El::Matrix<DataType, Device>& global_buffer =
      context.get_workspace_matrix(
        "allgather_recv_buffer",
        is_buffer_needed.second ? global_buffer_size : 0,
        1);
    kfac::allgather_blocks(
      buffers, local_buffer, global_buffer, &comm, allgather_mode);
  }
  prof_region_end("kfac-allgather", prof_sync);

#ifdef LBANN_NVPROF
  prof_region_begin("kfac-allgather-barrier", prof_color, prof_sync);
  CHECK_CUDA(cudaDeviceSynchronize());
  comm.trainer_barrier();
  prof_region_end("kfac-allgather-barrier", prof_sync);
#endif // LBANN_NVPROF

  prof_region_end("kfac-step", prof_sync);

  if(is_first_step) {
    for(auto& block : context.m_blocks) {
      for(auto& info : block->get_internal_matrix_info()) {
        std::ostringstream oss;
        oss << "K-FAC matrix allocation (rank="
            << comm.get_rank_in_trainer()
            << "): " << block->get_name()
            << " " << std::get<0>(info)
            << " (" << std::get<1>(info)
            << "x" << std::get<2>(info)
            << ")" << std::endl;
        std::cout << oss.str();
      }
    }
  }

}

} // namespace lbann

template <>
std::unique_ptr<lbann::KFAC> lbann::make<lbann::KFAC>(
  google::protobuf::Message const& msg_in)
{
  using AlgoType = lbann::KFAC;
  auto const& params =
    dynamic_cast<lbann_data::TrainingAlgorithm const&>(msg_in);

  lbann_data::KFAC kfac_params;
  LBANN_ASSERT(params.parameters().UnpackTo(&kfac_params));

  // SGD parameters
  auto const& sgd_params = kfac_params.sgd();
  auto const& stopping_criteria = sgd_params.stopping_criteria();
  std::unique_ptr<lbann::sgd_termination_criteria> stopping;
  switch (stopping_criteria.criterion_case()) {
  case lbann_data::SGD::TerminationCriteria::kMaxBatches:
    stopping = lbann::make_unique<lbann::batch_termination_criteria>(
      stopping_criteria.max_batches());
    break;
  case lbann_data::SGD::TerminationCriteria::kMaxEpochs:
    stopping = lbann::make_unique<lbann::epoch_termination_criteria>(
      stopping_criteria.max_epochs());
    break;
  case lbann_data::SGD::TerminationCriteria::kMaxSeconds:
    stopping = lbann::make_unique<lbann::seconds_termination_criteria>(
      stopping_criteria.max_seconds());
    //LBANN_ERROR("Time-based training not yet supported in SGD.");
    break;
  default:
    LBANN_ERROR("No stopping criteria specified.");
  }

  const auto parse_damping_params =
      [](const std::string str) {
        if(str == "")
          return std::vector<double>({AlgoType::damping_0_default});
        else {
          const auto ret = parse_list<double>(str);
          if(ret.size() > 2)
            LBANN_ERROR("The length of damping vectors should be 1 or 2.");
          return ret;
        }
      };

  const auto parse_update_intervals =
      [](const std::string str) {
        if(str == "")
          return std::vector<size_t>({1});
        else {
          const auto ret = parse_list<size_t>(str);
          if(ret.size() > 2)
            LBANN_ERROR("The length of update interval vectors should be 1 or 2.");
          return ret;
        }
      };

  const std::vector<double> damping_act_params = parse_damping_params(kfac_params.damping_act());
  const std::vector<double> damping_err_params = parse_damping_params(kfac_params.damping_err());
  const std::vector<double> damping_bn_act_params = parse_damping_params(kfac_params.damping_bn_act());
  const std::vector<double> damping_bn_err_params = parse_damping_params(kfac_params.damping_bn_err());
  size_t damping_warmup_steps = kfac_params.damping_warmup_steps();
  if(damping_warmup_steps == 0) damping_warmup_steps = AlgoType::damping_warmup_steps_default;
  double kronecker_decay = kfac_params.kronecker_decay();
  if(kronecker_decay == 0.0)
    kronecker_decay = AlgoType::kronecker_decay_default;
  const bool print_time = kfac_params.print_time();
  const bool print_matrix = kfac_params.print_matrix();
  const bool print_matrix_summary = kfac_params.print_matrix_summary();
  const bool use_pi = kfac_params.use_pi();
  const std::vector<size_t> update_intervals = parse_update_intervals(kfac_params.update_intervals());
  const size_t update_interval_steps = kfac_params.update_interval_steps();

  const std::string inverse_strategy_str = kfac_params.inverse_strategy();
  kfac::kfac_inverse_strategy inverse_strategy;
  if(inverse_strategy_str == "" || inverse_strategy_str == "all")
    inverse_strategy = kfac::kfac_inverse_strategy::ALL;
  else if(inverse_strategy_str == "each")
    inverse_strategy = kfac::kfac_inverse_strategy::EACH;
  else if(inverse_strategy_str == "root")
    inverse_strategy = kfac::kfac_inverse_strategy::ROOT;
  else {
    std::stringstream err;
    err << "Invalid inverse strategy type: "
        << inverse_strategy_str;
    LBANN_ERROR(err.str());
  }

  const std::vector<std::string> disable_layers =
      parse_list<std::string>(kfac_params.disable_layers());

  double learning_rate_factor = kfac_params.learning_rate_factor();
  double learning_rate_factor_gru = kfac_params.learning_rate_factor_gru();
  if(learning_rate_factor == 0.0)
    learning_rate_factor = 1.0;
  if(learning_rate_factor_gru == 0.0)
    learning_rate_factor_gru = learning_rate_factor;

  return make_unique<AlgoType>(
    params.name(),
    std::move(stopping),
    std::move(damping_act_params),
    std::move(damping_err_params),
    std::move(damping_bn_act_params),
    std::move(damping_bn_err_params),
    damping_warmup_steps,
    kronecker_decay,
    print_time,
    print_matrix,
    print_matrix_summary,
    use_pi,
    std::move(update_intervals),
    update_interval_steps,
    inverse_strategy,
    std::move(disable_layers),
    learning_rate_factor,
    learning_rate_factor_gru);

}
