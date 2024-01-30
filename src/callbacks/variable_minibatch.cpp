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
//
// lbann_variable_minibatch .hpp .cpp - Callback for variable-size mini-batches
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/variable_minibatch.hpp"

#include "lbann/data_ingestion/data_coordinator.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/io/input_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/exception.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <iostream>
#include <utility>

namespace lbann {
namespace callback {

variable_minibatch::variable_minibatch(uint64_t starting_mbsize)
  : m_starting_mbsize(starting_mbsize),
    m_current_mini_batch_size(starting_mbsize)
{}

void variable_minibatch::on_train_begin(model* m)
{
  // Avoid issues with the train method being called multiple times.
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  if (c.get_epoch() != 0) {
    return;
  }
  const auto& t = get_const_trainer();

  // Get first input layer in model
  input_layer<DataType>* input = nullptr;
  for (auto&& l : m->get_layers()) {
    input = dynamic_cast<input_layer<DataType>*>(l);
    if (input != nullptr) {
      break;
    }
  }
  if (input == nullptr) {
    LBANN_ERROR("could not get input layer");
  }

  if (m_starting_mbsize > t.get_max_mini_batch_size()) {
    throw lbann_exception(
      "variable_minibatch: starting mini-batch size is larger than max");
  }
  if (m_starting_mbsize == t.get_max_mini_batch_size()) {
    if (m->get_comm()->am_world_master()) {
      std::cout << "WARNING: starting mini-batch size equals max mini-batch "
                << "size and using variable-sized mini-batches" << std::endl;
    }
  }
  get_trainer().get_data_coordinator().calculate_num_iterations_per_epoch(
    m_starting_mbsize);
}

void variable_minibatch::on_epoch_end(model* m)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  const auto& t = get_const_trainer();

  // Get first input layer in model
  input_layer<DataType>* input = nullptr;
  for (auto&& l : m->get_layers()) {
    input = dynamic_cast<input_layer<DataType>*>(l);
    if (input != nullptr) {
      break;
    }
  }
  if (input == nullptr) {
    LBANN_ERROR("could not get input layer");
  }

  lbann_comm* comm = m->get_comm();
  uint64_t new_mbsize = 0;
  float new_lr = 0.0f;
  uint64_t ramp_time = 0;
  if (schedule(m, new_mbsize, new_lr, ramp_time)) {
    if (new_mbsize > t.get_max_mini_batch_size()) {
      if (comm->am_trainer_master()) {
        std::cout << "Model " << comm->get_trainer_rank() << ": WARNING "
                  << "requested new mini-batch size " << new_mbsize
                  << " is greater than the model maximum mini-batch size "
                  << t.get_max_mini_batch_size() << std::endl;
      }
      new_mbsize = t.get_max_mini_batch_size();
    }
    get_trainer().get_data_coordinator().calculate_num_iterations_per_epoch(
      new_mbsize);
    m_current_mini_batch_size = new_mbsize;
    m_ramp_count = ramp_time;
    if (new_lr != 0.0f) {
      if (ramp_time == 0) {
        // Change learning rate immediately.
        change_learning_rate(m, new_lr);
      }
      else {
        // Compute the per-epoch learning rate increment.
        float cur_lr = get_current_learning_rate(m);
        m_lr_incr = (new_lr - cur_lr) / ramp_time;
      }
      if (comm->am_trainer_master()) {
        std::cout << "Model " << comm->get_trainer_rank()
                  << ": Changing mini-batch size to " << new_mbsize
                  << " and learning rate to " << new_lr << " at epoch "
                  << c.get_epoch() << std::endl;
      }
    }
    else if (comm->am_trainer_master()) {
      std::cout << "Model " << comm->get_trainer_rank()
                << ": Changing mini-batch size to " << new_mbsize
                << " at epoch " << c.get_epoch() << std::endl;
    }
  }
  // Ramp the learning rate, if needed.
  if (m_ramp_count > 0) {
    --m_ramp_count;
    float target_lr = get_current_learning_rate(m) + m_lr_incr;
    change_learning_rate(m, target_lr);
    if (comm->am_trainer_master()) {
      std::cout << "Model " << comm->get_trainer_rank()
                << ": Variable-size mini-batch ramping learning rate to "
                << target_lr << std::endl;
    }
  }
}

void variable_minibatch::change_learning_rate(model* m, float new_lr) const
{
  for (weights* w : m->get_weights()) {
    if (optimizer* opt = w->get_optimizer()) {
      auto& dt_opt = dynamic_cast<data_type_optimizer<DataType>&>(*opt);
      dt_opt.set_learning_rate(new_lr);
    }
  }
}

float variable_minibatch::get_current_learning_rate(model* m) const
{
  for (weights* w : m->get_weights()) {
    if (optimizer* opt = w->get_optimizer()) {
      auto& dt_opt = dynamic_cast<data_type_optimizer<DataType> const&>(*opt);
      return dt_opt.get_learning_rate();
    }
  }
  return 0.0f;
}

step_minibatch::step_minibatch(uint64_t starting_mbsize,
                               uint64_t step,
                               uint64_t ramp_time)
  : variable_minibatch(starting_mbsize), m_step(step), m_ramp_time(ramp_time)
{}

bool step_minibatch::schedule(model* m,
                              uint64_t& new_mbsize,
                              float& new_lr,
                              uint64_t& ramp_time)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  if (c.get_epoch() % m_step == 0) {
    new_mbsize = m_current_mini_batch_size * 2;
    new_lr = get_current_learning_rate(m) * 2;
    ramp_time = m_ramp_time;
    return true;
  }
  else {
    return false;
  }
}

void step_minibatch::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_step_minibatch();
  msg->set_starting_mbsize(m_starting_mbsize);
  msg->set_step(m_step);
  msg->set_ramp_time(m_ramp_time);
}

minibatch_schedule::minibatch_schedule(uint64_t starting_mbsize,
                                       std::vector<minibatch_step> steps)
  : variable_minibatch(starting_mbsize), m_steps(std::move(steps))
{
  std::sort(m_steps.rbegin(),
            m_steps.rend(),
            [](const minibatch_step& a, const minibatch_step& b) {
              return a.epoch < b.epoch;
            });
}

bool minibatch_schedule::schedule(model* m,
                                  uint64_t& new_mbsize,
                                  float& new_lr,
                                  uint64_t& ramp_time)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  if (!m_steps.empty() && c.get_epoch() == m_steps.back().epoch) {
    new_mbsize = m_steps.back().mbsize;
    new_lr = m_steps.back().lr;
    ramp_time = m_steps.back().ramp_time;
    m_steps.pop_back();
    return true;
  }
  return false;
}

void minibatch_schedule::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_minibatch_schedule();
  msg->set_starting_mbsize(m_starting_mbsize);
  for (auto const& step : m_steps) {
    auto* sched_step = msg->add_step();
    sched_step->set_epoch(step.epoch);
    sched_step->set_mbsize(step.mbsize);
    sched_step->set_lr(step.lr);
    sched_step->set_ramp_time(step.ramp_time);
  }
}

std::unique_ptr<callback_base> build_step_minibatch_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  std::shared_ptr<lbann_summary> const&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackStepMinibatch&>(proto_msg);
  return std::make_unique<step_minibatch>(params.starting_mbsize(),
                                          params.step(),
                                          params.ramp_time());
}

std::unique_ptr<callback_base> build_minibatch_schedule_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  std::shared_ptr<lbann_summary> const&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackMinibatchSchedule&>(
      proto_msg);
  std::vector<minibatch_schedule::minibatch_step> steps;
  for (int i = 0; i < params.step_size(); ++i) {
    const auto& proto_step = params.step(i);
    steps.emplace_back(proto_step.epoch(),
                       proto_step.mbsize(),
                       proto_step.lr(),
                       proto_step.ramp_time());
  }
  return std::make_unique<minibatch_schedule>(params.starting_mbsize(), steps);
}

} // namespace callback
} // namespace lbann
