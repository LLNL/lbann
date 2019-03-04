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

#include "lbann/proto/factories.hpp"
#include "lbann/objective_functions/layer_term.hpp"

namespace lbann {
namespace proto {

namespace {

/** Instantiate a model based on prototext. */
trainer* instantiate_trainer(lbann_comm* comm,
                           objective_function* obj,
                           const lbann_data::Optimizer& proto_opt,
                           const lbann_data::Model& proto_model) {
  std::stringstream err;

  // Default optimizer
  auto&& opt = construct_optimizer(comm, proto_opt);

  // Construct model
  const auto& type = proto_model.type();
  const auto& mini_batch_size = proto_model.mini_batch_size();
  return new trainer(comm, mini_batch_size, obj, opt);
}

} // namespace

model* construct_trainer(lbann_comm* comm,
                         const lbann_data::Trainer& proto_trainer,
                         const lbann_data::Optimizer& proto_opt,
                         const lbann_data::TrainingAlgorithm& proto_training_alg) {

#if 0
  /// BVE TODO FIXME this should be done somewhere else
  // Construct objective function
  const auto& proto_obj = proto_model.objective_function();
  auto&& obj = construct_objective_function(proto_obj);
#endif

  // Construct callbacks
  std::vector<lbann_callback*> callback_list;
  auto&& summarizer = construct_summarizer(comm, proto_trainer);
  for (int i=0; i<proto_model.callback_size(); i++) {
    callback_list.push_back(construct_callback(comm,
                                               proto_trainer.callback(i),
                                               data_readers,
                                               summarizer));
  }

  // Instantiate trainer
  auto&& t = instantiate_trainer(comm, obj, proto_opt, proto_trainer);
  for (auto&& cb  : callback_list) { t->add_callback(cb); }
  const auto& name = proto_trainer.name();
  if (!name.empty()) {
    t->set_name(name);
  }
  // for (auto t : data_readers) {
  //   t.second->set_model(m);
  // }
  return t;

}

} // namespace proto
} // namespace lbann
