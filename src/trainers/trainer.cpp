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

#include "lbann/trainers/trainer.hpp"
#include "lbann/callbacks/callback.hpp"
//#include "lbann/callbacks/callback_save_model.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/layers/transform/dummy.hpp"
#include "lbann/layers/transform/split.hpp"
#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/objective_functions/layer_term.hpp"
#include "lbann/metrics/layer_metric.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"
#include "lbann/training_algorithms/sgd_training_algorithm.hpp"
#include <string>
#include <unistd.h>
#include <iomanip>
#include <queue>
#include <unordered_set>
#include <lbann.pb.h>

#include "mpi.h"

namespace lbann {

////////////////////////////////////////////////////////////
// Constructors and destructor
////////////////////////////////////////////////////////////

trainer::trainer(lbann_comm *comm)
  : m_comm(comm),
    m_io_thread_pool(),
    m_background_io_allowed(true) {

  // Default trainer name
  m_name = "trainer" + std::to_string(m_comm->get_trainer_rank());
}

trainer::trainer(const trainer& other) :
  m_comm(other.m_comm),
  m_background_io_allowed(other.m_background_io_allowed) {

  // Deep copies
  // m_io_thread_pool = (other.m_io_thread_pool ?
  //                     other.m_io_thread_pool->copy() : nullptr);
}

trainer& trainer::operator=(const trainer& other) {

  // Shallow copies
  m_comm = other.m_comm;
  m_background_io_allowed = other.m_background_io_allowed;

  // Deep copies
  // m_io_thread_pool = (other.m_io_thread_pool ?
  //                     other.m_io_thread_pool->copy() : nullptr);

  return *this;
}

trainer::~trainer() {
}

////////////////////////////////////////////////////////////
// Trainer specification
////////////////////////////////////////////////////////////

void trainer::set_name(std::string const& name) {
  m_name = name;
}

description trainer::get_description() const {

  // Construct description object
  description desc(get_name());
  desc.add("Background I/O", m_background_io_allowed);

  // Result
  return desc;

}

////////////////////////////////////////////////////////////
// Setup
////////////////////////////////////////////////////////////

void trainer::setup(std::unique_ptr<thread_pool> io_thread_pool) {
  // Setup I/O threads - set up before setting up the layers (input
  // layer depends on having a properly initialized thread pool)
  m_io_thread_pool = std::move(io_thread_pool);
}

/// Check if there is already an execution context for the model in this mode, if not create one
trainer::execution_context_key_pair_t trainer::check_and_build_execution_context(training_algorithm& alg,
                                                                                 observer_ptr<model> model,
                                                                                 execution_mode mode) {
  auto key = std::make_pair(model,mode);
  if(m_model_execution_context.count(key) == 0) {
    /// Create a execution context for each model and execution mode
    std::unique_ptr<execution_context> context;
    if(dynamic_cast<observer_ptr<sgd_training_algorithm>>(&alg) != nullptr) {
      /// @todo BVE FIXME Figure out how to get a good mini-batch size
      /// in here
      context = make_unique<sgd_execution_context>(this, m_comm, mode, model->get_max_mini_batch_size());
    }else {
      context = make_unique<execution_context>(this, m_comm, mode);
    }
    m_model_execution_context.emplace(key,std::move(context));
  }
  return key;
}

/// Check if there is already an execution context for the model in this mode, if not create one
trainer::execution_context_key_pair_t trainer::check_and_build_execution_context(const execution_context& c,
                                                                                 model& model,
                                                                                 execution_mode mode) {
  auto key = std::make_pair(&model, mode);
  if(m_model_execution_context.count(key) == 0) {
    std::unique_ptr<execution_context> context;
    if(dynamic_cast<observer_ptr<const sgd_execution_context>>(&c) != nullptr) {
      context = make_unique<sgd_execution_context>(this, m_comm, mode, model.get_max_mini_batch_size());
    }else {
      context = make_unique<execution_context>(this, m_comm, mode);
    }
    m_model_execution_context.emplace(key,std::move(context));
  }
  return key;
}

execution_context& trainer::get_execution_context(observer_ptr<model> model,
                                                  execution_mode mode) {
  auto key = std::make_pair(model,mode);
  return get_execution_context(key);
}

execution_context& trainer::get_execution_context(execution_context_key_pair_t key) {
  if(m_model_execution_context.count(key) == 0) {
    LBANN_ERROR("No execution context for this model / mode pair");
  }
  return static_cast<sgd_execution_context&>(*(m_model_execution_context[key].get()));
}

void trainer::delete_execution_context(execution_context_key_pair_t key) {
  if(m_model_execution_context.count(key) == 0) {
    LBANN_WARNING("Attempting to delete an invalid execution context for model="
                  + (key.first)->get_name() + " / " + to_string(key.second));
  }
  m_model_execution_context.erase(key);
}

void trainer::for_each_execution_context(std::function<void(observer_ptr<execution_context>)>fn) {
  for(auto&& c : m_model_execution_context) {
    // auto&& model = c.first.first;
    // auto&& mode = c.first.second;
    auto&& context = c.second;
    fn(context.get());
  }
}


////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////
void trainer::apply(training_algorithm& alg,
                    observer_ptr<model> model,
                    execution_mode mode,
                    termination_criteria const& term_criteria) {

  auto key = check_and_build_execution_context(alg, model, mode);

  /// Apply the training algorithm to train the model
  alg.apply(*(m_model_execution_context[key].get()), *model, mode, term_criteria);
}

void trainer::train(observer_ptr<model> model, El::Int num_epochs, El::Int num_batches) {
  auto sgd = make_unique<sgd_training_algorithm>();
  auto key = check_and_build_execution_context(*sgd.get(), model, execution_mode::training);
  /// Apply the training algorithm to train the model
  sgd.get()->train(static_cast<sgd_execution_context&>(*(m_model_execution_context[key].get())), *model, num_epochs, num_batches);
}

void trainer::evaluate(observer_ptr<model> model, execution_mode mode, El::Int num_batches) {
  auto sgd = make_unique<sgd_training_algorithm>();
  auto key = check_and_build_execution_context(*sgd.get(), model, mode);
  /// Apply the training algorithm to evaluate the model
  sgd.get()->evaluate(static_cast<sgd_execution_context&>(*(m_model_execution_context[key].get())), *model, mode, num_batches);
}

}  // namespace lbann
