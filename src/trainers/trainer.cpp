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

trainer::trainer(lbann_comm *comm,
                 size_t mini_batch_size,
                 std::map<execution_mode,
                 generic_data_reader *> data_readers)
  : m_comm(comm),
    m_max_mini_batch_size(mini_batch_size),
    m_io_thread_pool(),
    m_background_io_allowed(true),
    m_data_coordinator(comm, data_readers) {

  // Default trainer name
  m_name = "trainer" + std::to_string(m_comm->get_trainer_rank());
}

trainer::trainer(const trainer& other) :
  m_comm(other.m_comm),
  m_max_mini_batch_size(other.m_max_mini_batch_size),
  m_background_io_allowed(other.m_background_io_allowed),
  m_data_coordinator(other.m_data_coordinator) {

  // Deep copies
  // m_io_thread_pool = (other.m_io_thread_pool ?
  //                     other.m_io_thread_pool->copy() : nullptr);
  m_callbacks.reserve(other.m_callbacks.size());
  for (auto const& cb : other.m_callbacks) {
    m_callbacks.emplace_back(cb->copy());
  }
}

trainer& trainer::operator=(const trainer& other) {

  // Shallow copies
  m_comm = other.m_comm;
  m_max_mini_batch_size = other.m_max_mini_batch_size;
  m_background_io_allowed = other.m_background_io_allowed;

  // Deep copies
  // m_io_thread_pool = (other.m_io_thread_pool ?
  //                     other.m_io_thread_pool->copy() : nullptr);
  m_callbacks.reserve(other.m_callbacks.size());
  for (auto const& cb : other.m_callbacks) {
    m_callbacks.emplace_back(cb->copy());
  }

  return *this;
}

trainer::~trainer() {
}

////////////////////////////////////////////////////////////
// Trainer specification
////////////////////////////////////////////////////////////

void trainer::set_name(std::string const& name) {
  if (name.empty()) {
    LBANN_ERROR("attempted to rename trainer \"", get_name(), "\" with empty string");
  }
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

  // Set up callbacks
  for (auto& cb : m_callbacks) {
    cb->setup(this);
  }

  m_data_coordinator.setup(get_max_mini_batch_size());
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
      context = make_unique<sgd_execution_context>(*this, alg, m_comm, mode, get_max_mini_batch_size());
    }else {
      context = make_unique<execution_context>(*this, alg, m_comm, mode);
    }
    m_model_execution_context.emplace(key,std::move(context));
  }
  return key;
}

/// Check if there is already an execution context for the model in this mode, if not create one
trainer::execution_context_key_pair_t trainer::check_and_build_execution_context(execution_context& c,
                                                                                 model& model,
                                                                                 execution_mode mode) {
  auto key = std::make_pair(&model, mode);
  if(m_model_execution_context.count(key) == 0) {
    std::unique_ptr<execution_context> context;
    //    observer_ptr<training_algorithm> alg = const_cast
    if(dynamic_cast<observer_ptr</*const */sgd_execution_context>>(&c) != nullptr) {
      context = make_unique<sgd_execution_context>(*this, c.get_training_algorithm(), m_comm, mode, get_max_mini_batch_size());
    }else {
      context = make_unique<execution_context>(*this, c.get_training_algorithm(), m_comm, mode);
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

  /// @todo BVE FIXME seems like there is a bug here about mapping
  /// execution contexts to the right model
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
  TargetModeDimMap data_dimensions_map = get_data_coordinator().get_data_dims();
  for(data_reader_target_mode m : data_reader_target_mode_iterator()) {
    std::cout << "data_dimensions_map[" << to_string(m) << "]=";
    for(auto v : data_dimensions_map[m]) {
      std::cout << " " << v;
    }
    std::cout << std::endl;
  }
  alg.setup_models({model}, get_max_mini_batch_size(), data_dimensions_map);
  /// Apply the training algorithm to train the model
  alg.apply(*(m_model_execution_context[key].get()), *model, get_data_coordinator(), mode, term_criteria);
}

void trainer::train(observer_ptr<model> model, El::Int num_epochs, El::Int num_batches) {
  auto sgd = make_unique<sgd_training_algorithm>();
  auto key = check_and_build_execution_context(*sgd.get(), model, execution_mode::training);
  TargetModeDimMap data_dimensions_map = get_data_coordinator().get_data_dims();
  for(data_reader_target_mode m : data_reader_target_mode_iterator()) {
    std::cout << "data_dimensions_map[" << to_string(m) << "]=";
    for(auto v : data_dimensions_map[m]) {
      std::cout << " " << v;
    }
    std::cout << std::endl;
  }
  sgd.get()->setup_models({model}, get_max_mini_batch_size(), data_dimensions_map);
  /// Apply the training algorithm to train the model
  sgd.get()->train(static_cast<sgd_execution_context&>(*(m_model_execution_context[key].get())), *model, get_data_coordinator(), num_epochs, num_batches);
}

void trainer::evaluate(observer_ptr<model> model, execution_mode mode, El::Int num_batches) {
  auto sgd = make_unique<sgd_training_algorithm>();
  auto key = check_and_build_execution_context(*sgd.get(), model, mode);
  TargetModeDimMap data_dimensions_map = get_data_coordinator().get_data_dims();
  for(data_reader_target_mode m : data_reader_target_mode_iterator()) {
    std::cout << "data_dimensions_map[" << to_string(m) << "]=";
    for(auto v : data_dimensions_map[m]) {
      std::cout << " " << v;
    }
    std::cout << std::endl;
  }
  sgd.get()->setup_models({model}, get_max_mini_batch_size(), data_dimensions_map);
  /// Apply the training algorithm to evaluate the model
  sgd.get()->evaluate(static_cast<sgd_execution_context&>(*(m_model_execution_context[key].get())), *model, get_data_coordinator(), mode, num_batches);
}

// =============================================
// Checkpointing
// =============================================

bool trainer::save_to_checkpoint_shared() {
  auto save_checkpoint = [this](observer_ptr<execution_context> ctx) {
    ctx->save_to_checkpoint_shared(this->get_persist_obj());
  };
  for_each_execution_context(save_checkpoint);
  save_rng_to_checkpoint_shared(get_persist_obj(), m_comm);

  if (m_comm->am_trainer_master()) {
    write_cereal_archive(*this, get_persist_obj(), "trainer.xml");
  }
  return true;
}

bool trainer::load_from_checkpoint_shared(persist& p) {
  try {
    load_from_shared_cereal_archive(*this, p, *get_comm(), "trainer.xml");
  }catch (NonexistentArchiveFile const& e) {
    LBANN_MSG(e.what());
    return false;
  }
  return true;
}

bool trainer::load_from_checkpoint_shared(model& m, execution_context& c) {
  load_rng_from_checkpoint(get_persist_obj(), m_comm);

  auto flag = get_data_coordinator().load_from_checkpoint_shared(m_persist);

  execution_mode current_mode = c.get_execution_mode();

  for(execution_mode mode : execution_mode_iterator()) {
    /// Restart should optionally load any other valid contexts
    if(mode == execution_mode::invalid) { continue; }
    trainer::execution_context_key_pair_t key;
    try {
      if(current_mode == mode) {
        /// Restart has to be able to load the currently running execution context
        c.load_from_checkpoint_shared(get_persist_obj());
      }else {
        key = check_and_build_execution_context(c, m, mode);
        auto& evaluation_context = static_cast<sgd_execution_context&>(get_execution_context(key));
        evaluation_context.load_from_checkpoint_shared(get_persist_obj());
      }
    }catch (NonexistentArchiveFile const&) {
      // Ignore the exception if the file is not for the current execution mode
      if(current_mode == mode) {
        LBANN_ERROR("Failed to restart model, invalid execution mode: " + to_string(current_mode));
      }else {
        delete_execution_context(key);
      }
    }
  }
  return flag;
}

bool trainer::save_to_checkpoint_distributed(){
  auto save_checkpoint = [this](observer_ptr<execution_context> ctx) {
    ctx->save_to_checkpoint_distributed(this->get_persist_obj());
  };
  for_each_execution_context(save_checkpoint);
  save_rng_to_checkpoint_distributed(get_persist_obj(), m_comm);
  return true;
}

bool trainer::load_from_checkpoint_distributed(persist& p){
  read_cereal_archive(*this, p, "trainer.xml");
  return false;
}

bool trainer::load_from_checkpoint_distributed(model& m, execution_context& c){
  load_rng_from_checkpoint(get_persist_obj(), m_comm);

  execution_mode current_mode = c.get_execution_mode();

  for(execution_mode mode : execution_mode_iterator()) {
    /// Restart should optionally load any other valid contexts
    if(mode == execution_mode::invalid) { continue; }
    trainer::execution_context_key_pair_t key;
    try {
      if(current_mode == mode) {
        /// Restart has to be able to load the currently running  execution context
        c.load_from_checkpoint_distributed(get_persist_obj());
      }else {
        key = check_and_build_execution_context(c, m, mode);
        auto& evaluation_context = static_cast<sgd_execution_context&>(get_execution_context(key));
        evaluation_context.load_from_checkpoint_distributed(get_persist_obj());
      }
    }catch (NonexistentArchiveFile const&) {
      // Ignore the exception if the file is not for the current execution mode
      if(current_mode == mode) {
        LBANN_ERROR("Failed to restart model, invalid execution mode: " + to_string(current_mode));
      }else {
        delete_execution_context(key);
      }
    }
  }
  return true;
}

void trainer::write_proto(lbann_data::Trainer* proto) {
  proto->Clear();
  if (m_comm->am_world_master()) {
    proto->set_mini_batch_size(m_max_mini_batch_size);
  }
}

}  // namespace lbann
