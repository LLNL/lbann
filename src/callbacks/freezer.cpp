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
// freezer .hpp .cpp - Callback hooks to time training
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/freezer.hpp"
#include <callbacks.pb.h>

#include <algorithm>
#include <string>
#include <unordered_map>

namespace lbann {
namespace callback {

freezer::freezer(freezer::freeze_t&& freeze_e, freezer::freeze_t&& unfreeze_e,
                 freezer::freeze_t&& freeze_s, freezer::freeze_t&& unfreeze_s)
: m_freeze_by_epoch(freeze_e), m_unfreeze_by_epoch(unfreeze_e),
  m_freeze_by_step(freeze_s), m_unfreeze_by_step(unfreeze_s) {
}

void freezer::on_epoch_begin(model *m) {
  const auto& c = dynamic_cast<sgd_execution_context&>(m->get_execution_context());
  const auto epoch = c.get_epoch();
  auto ret = m_freeze_by_epoch.equal_range(epoch);

  if (ret.first == ret.second) {
    return;
  }

  std::set<std::string> layer_names;
  for (freeze_t::const_iterator it = ret.first; it != ret.second; it++) {
    layer_names.insert(it->second);
  }

  for (auto lptr : m->get_layers()) {
    if (layer_names.count(lptr->get_name()) > 0ul) {
      lptr->freeze();
    }
  }
}

void freezer::on_epoch_end(model *m) {
  const auto& c = dynamic_cast<sgd_execution_context&>(m->get_execution_context());
  const auto epoch = c.get_epoch();
  auto ret = m_unfreeze_by_epoch.equal_range(epoch);

  if (ret.first == ret.second) {
    return;
  }

  std::set<std::string> layer_names;
  for (freeze_t::const_iterator it = ret.first; it != ret.second; it++) {
    layer_names.insert(it->second);
  }

  for (auto lptr : m->get_layers()) {
    if (layer_names.count(lptr->get_name()) > 0ul) {
      lptr->unfreeze();
    }
  }
}

void freezer::on_batch_begin(model *m) {
  const auto& c = m->get_execution_context();
  const auto step = c.get_step();
  auto ret = m_freeze_by_step.equal_range(step);

  if (ret.first == ret.second) {
    return;
  }

  std::set<std::string> layer_names;
  for (freeze_t::const_iterator it = ret.first; it != ret.second; it++) {
    layer_names.insert(it->second);
  }

  for (auto lptr : m->get_layers()) {
    if (layer_names.count(lptr->get_name()) > 0ul) {
      lptr->freeze();
    }
  }
}

void freezer::on_batch_end(model *m) {
  const auto& c = m->get_execution_context();
  const auto step = c.get_step();
  auto ret = m_unfreeze_by_step.equal_range(step);

  if (ret.first == ret.second) {
    return;
  }

  std::set<std::string> layer_names;
  for (freeze_t::const_iterator it = ret.first; it != ret.second; it++) {
    layer_names.insert(it->second);
  }

  for (auto lptr : m->get_layers()) {
    if (layer_names.count(lptr->get_name()) > 0ul) {
      lptr->unfreeze();
    }
  }
}

std::unique_ptr<callback_base>
build_freezer_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackFreezer&>(proto_msg);

  freezer::freeze_t freeze_by_epoch;
  freezer::freeze_t unfreeze_by_epoch;
  freezer::freeze_t freeze_by_step;
  freezer::freeze_t unfreeze_by_step;
  const int num_freeze_by_epoch = params.freeze_by_epoch_size();
  for (int i=0; i < num_freeze_by_epoch; ++i) {
    freeze_by_epoch.insert(std::make_pair(params.freeze_by_epoch(i).epoch(),
                                          params.freeze_by_epoch(i).layer()));
  }
  const int num_unfreeze_by_epoch = params.unfreeze_by_epoch_size();
  for (int i=0; i < num_unfreeze_by_epoch; ++i) {
    unfreeze_by_epoch.insert(std::make_pair(params.unfreeze_by_epoch(i).epoch(),
                                            params.unfreeze_by_epoch(i).layer()));
  }
  const int num_freeze_by_step = params.freeze_by_step_size();
  for (int i=0; i < num_freeze_by_step; ++i) {
    freeze_by_step.insert(std::make_pair(params.freeze_by_step(i).step(),
                                         params.freeze_by_step(i).layer()));
  }
  const int num_unfreeze_by_step = params.unfreeze_by_step_size();
  for (int i=0; i < num_unfreeze_by_step; ++i) {
    unfreeze_by_step.insert(std::make_pair(params.unfreeze_by_step(i).step(),
                                           params.unfreeze_by_step(i).layer()));
  }

  return make_unique<freezer>(std::move(freeze_by_epoch),
                              std::move(unfreeze_by_epoch),
                              std::move(freeze_by_step),
                              std::move(unfreeze_by_step));
}

} // namespace callback
} // namespace lbann
