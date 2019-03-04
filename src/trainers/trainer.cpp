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
             int mini_batch_size)
  : m_comm(comm),
    m_io_thread_pool(),
    m_background_io_allowed(true) {

  // Default trainer name
  static El::Int num_trainers = 0;
  m_name = "trainer" + std::to_string(num_trainers);
  num_trainers++;
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

void trainer::add_callback(lbann_callback *cb) {
  if (cb == nullptr) {
    throw lbann_exception("trainer: Attempted to add null pointer as a callback.");
  }
  m_callbacks.push_back(cb);
}

void trainer::set_name(std::string name) {
  m_name = name;
}

description trainer::get_description() const {

  // Construct description object
  description desc(get_name());
  desc.add("Type", get_type());

  /// @todo Descriptions for objective function, metrics, callbacks

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
std::pair<observing_ptr<model>, execution_mode> trainer::check_and_build_execution_context(observing_ptr<training_algorithm> alg,
                                                                                           observing_ptr<model> model,
                                                                                           execution_mode mode) {
  auto key = std::make_pair(model,mode);
  if(m_model_execution_context.count(key) == 0) {
    /// Create a execution context for each model and execution mode
    auto *sgd_alg = (observing_ptr<sgd_training_algorithm>) dynamic_cast<observing_ptr<sgd_training_algorithm>>(alg);
    std::unique_ptr<execution_context> context;
    if(sgd_alg != nullptr) {
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

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////
void trainer::apply(observing_ptr<training_algorithm> alg,
                    observing_ptr<model> model,
                    execution_mode mode,
                    termination_criteria const& term_criteria) {

  auto key = check_and_build_execution_context(alg, model, mode);

  /// Apply the training algorithm to train the model
  alg->apply(*(m_model_execution_context[key].get()), *model, mode, term_criteria);
}

void trainer::train(observing_ptr<model> model, El::Int num_epochs, El::Int num_batches) {
  auto sgd = make_unique<sgd_training_algorithm>();
  auto key = check_and_build_execution_context(sgd.get(), model, execution_mode::training);
  /// Apply the training algorithm to train the model
  sgd.get()->train(static_cast<sgd_execution_context&>(*(m_model_execution_context[key].get())), *model, num_epochs, num_batches);
}

  void trainer::evaluate(observing_ptr<model> model, execution_mode mode, El::Int num_batches) {
  auto sgd = make_unique<sgd_training_algorithm>();
  auto key = check_and_build_execution_context(sgd.get(), model, mode);
  /// Apply the training algorithm to evaluate the model
  sgd.get()->evaluate(static_cast<sgd_execution_context&>(*(m_model_execution_context[key].get())), *model, mode, num_batches);
}

#if 0
//this is for data store functionality
void trainer::collect_indices(execution_mode mode) {
  reset_mode_and_model(mode);
  while (true) {
    get_layer(0).forward_prop();
    bool finished = true;
    finished = get_layer(0).update() && finished;
    if (finished) {
      break;
    }
  }
  //this may not be necessary, but shouldn't hurt
  reset_epoch_statistics(mode);
}

////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

/* struct used to serialize mode fields in file and MPI transfer */
struct lbann_trainer_header {
  uint32_t execution_mode;
  uint32_t terminate_training;
  uint64_t epoch;
  uint64_t training_step;
  uint64_t validation_step;
  uint64_t testing_step;
  uint32_t max_mini_batch_size;
  uint32_t current_mini_batch_size;
  uint32_t current_phase;
  uint32_t callback_type;;
};

bool trainer::save_to_checkpoint_shared(persist& p) {
  // write out fields we need to save for trainer
  if (p.get_cb_type() != callback_type::validation) {
    if (m_comm->am_trainer_master()) {
      p.write_uint32(persist_type::train, "execution_mode",     (uint32_t) m_execution_mode);
      p.write_uint32(persist_type::train, "terminate_training", (uint32_t) m_terminate_training);
      p.write_uint64(persist_type::train, "epoch",              (uint64_t) m_epoch);
      p.write_uint64(persist_type::train, "training_step",      (uint64_t) get_step(execution_mode::training));
      p.write_uint64(persist_type::train, "testing_step",       (uint64_t) get_step(execution_mode::testing));
      p.write_uint32(persist_type::train, "max_mini_batch_size",      (uint32_t) m_max_mini_batch_size);
      p.write_uint32(persist_type::train, "current_mini_batch_size",      (uint32_t) m_current_mini_batch_size);
      p.write_uint32(persist_type::train, "current_phase",      (uint32_t) m_current_phase);
      p.write_uint32(persist_type::train, "persist_callback_type",      (uint32_t) p.get_cb_type());
      /// @todo BVE FIXME
      if(p.get_cb_type() == callback_type::batch) {
        p.write_uint64(persist_type::validate, "validation_step",       (uint64_t) get_step(execution_mode::validation));
      }
    }

  }
  else{
    if (m_comm->am_trainer_master()) {
      p.write_uint64(persist_type::validate, "validation_step",       (uint64_t) get_step(execution_mode::validation));
    }
    save_rng_to_checkpoint_shared(p, m_comm);
  }
  return true;
}

bool trainer::load_from_checkpoint_shared(persist& p) {
  // have rank 0 read the file
  // read state from file
  struct lbann_trainer_header header;
  // Assume checkpoint reload from epoch end not step end
  if (m_comm->am_trainer_master()) {
    if (p.get_cb_type() != callback_type::validation) {
      p.read_uint32(persist_type::train, "execution_mode",     &header.execution_mode);
      p.read_uint32(persist_type::train, "terminate_training", &header.terminate_training);
      p.read_uint64(persist_type::train, "current_epoch",      &header.current_epoch);
      p.read_uint64(persist_type::train, "current_step",       &header.current_step);
      if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
        p.read_uint64(persist_type::validate, "current_validation_step",       &header.current_validation_step);
      p.read_uint64(persist_type::train, "current_testing_step",       &header.current_testing_step);
      p.read_uint32(persist_type::train, "max_mini_batch_size",      &header.max_mini_batch_size);
      p.read_uint32(persist_type::train, "current_mini_batch_size",      &header.current_mini_batch_size);
      p.read_uint32(persist_type::train, "current_phase",      &header.current_phase);
      p.read_uint32(persist_type::train, "persist_callback_type",     &header.callback_type);
    } else {
      p.read_uint64(persist_type::validate, "current_validation_step",       &header.current_validation_step);
    }
  }
  load_rng_from_checkpoint_shared(p, m_comm);
  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  m_comm->trainer_broadcast(0, header);
  // set our member params from values read from disk
  if (p.get_cb_type() != callback_type::validation) {
    m_execution_mode     = (execution_mode) header.execution_mode;
    m_terminate_training = (bool)           header.terminate_training;
    m_current_epoch      = (int)            header.current_epoch;
    m_current_step       = (int)            header.current_step;
    if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
      m_current_validation_step = (int)       header.current_validation_step;
    m_current_testing_step = (int)          header.current_testing_step;
    m_max_mini_batch_size = (int)           header.max_mini_batch_size;
    m_current_mini_batch_size = (int)       header.current_mini_batch_size;
    m_current_phase      =                  header.current_phase;
    // set state of persist object to know which type of ckpt we are returning from.
    p.set_cb_type((callback_type) header.callback_type);
  } else {
    m_current_validation_step = (int)       header.current_validation_step;
  }

  for (weights *w : m_weights) {
    w->load_from_checkpoint_shared(p);
  }

  // read in each layer
  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->load_from_checkpoint_shared(p)) {
      return false;
    }
  }
  if(get_num_iterations_per_epoch(execution_mode::validation) != 0){
    for (const auto& m : m_metrics) {
      m->load_from_checkpoint_shared(p);
    }
  }
#ifdef LBANN_HAS_GPU
  El::GPUManager::SynchronizeDevice();
#endif // LBANN_HAS_GPU
  return true;
}

bool trainer::save_to_checkpoint_distributed(persist& p){
  // write out fields we need to save for trainer
  if (p.get_cb_type() != callback_type::validation) {
    p.write_uint32(persist_type::train, "execution_mode",     (uint32_t) m_execution_mode);
    p.write_uint32(persist_type::train, "terminate_training", (uint32_t) m_terminate_training);
    p.write_uint64(persist_type::train, "current_epoch",      (uint64_t) m_current_epoch);
    p.write_uint64(persist_type::train, "current_step",       (uint64_t) m_current_step);
    p.write_uint64(persist_type::train, "current_testing_step",       (uint64_t) m_current_testing_step);
    p.write_uint32(persist_type::train, "max_mini_batch_size",      (uint32_t) m_max_mini_batch_size);
    p.write_uint32(persist_type::train, "current_mini_batch_size",      (uint32_t) m_current_mini_batch_size);
    p.write_uint32(persist_type::train, "current_phase",      (uint32_t) m_current_phase);
    p.write_uint32(persist_type::train, "persist_callback_type",      (uint32_t) p.get_cb_type());
    if(p.get_cb_type() == callback_type::batch)
      p.write_uint64(persist_type::validate, "current_validataion_step",       (uint64_t) m_current_validation_step);

    for (weights *w : m_weights) {
      w->save_to_checkpoint_distributed(p);
    }

    for (size_t l = 0; l < m_layers.size(); l++) {
      if (! m_layers[l]->save_to_checkpoint_distributed(p)) {
        return false;
      }
    }
    if(p.get_cb_type() == callback_type::batch || get_num_iterations_per_epoch(execution_mode::validation) == 0){
       save_rng_to_checkpoint_shared(p, m_comm);
      for (const auto& m : m_metrics) {
        m->save_to_checkpoint_distributed(p);
      }
    }
  }

  else {
    p.write_uint64(persist_type::validate, "current_validataion_step",       (uint64_t) m_current_validation_step);
    save_rng_to_checkpoint_shared(p, m_comm);

    for (size_t l = 0; l < m_layers.size(); l++) {
      if (! m_layers[l]->save_to_checkpoint_distributed(p)) {
        return false;
      }
    }
    for (const auto& m : m_metrics) {
      m->save_to_checkpoint_distributed(p);
    }
  }
  return true;
}

bool trainer::load_from_checkpoint_distributed(persist& p){
  struct lbann_trainer_header header;
  p.read_uint32(persist_type::train, "execution_mode",     &header.execution_mode);
  p.read_uint32(persist_type::train, "terminate_training", &header.terminate_training);
  p.read_uint64(persist_type::train, "current_epoch",      &header.current_epoch);
  p.read_uint64(persist_type::train, "current_step",       &header.current_step);
  if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
    p.read_uint64(persist_type::validate, "current_validation_step",       &header.current_validation_step);
  p.read_uint64(persist_type::train, "current_testing_step",       &header.current_testing_step);
  p.read_uint32(persist_type::train, "max_mini_batch_size",      &header.max_mini_batch_size);
  p.read_uint32(persist_type::train, "current_mini_batch_size",      &header.current_mini_batch_size);
  p.read_uint32(persist_type::train, "current_phase",      &header.current_phase);
  p.read_uint32(persist_type::train, "persist_callback_type",     &header.callback_type);

  m_execution_mode     = (execution_mode) header.execution_mode;
  m_terminate_training = (bool)           header.terminate_training;
  m_current_epoch      = (int)            header.current_epoch;
  m_current_step       = (int)            header.current_step;
  if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
    m_current_validation_step = (int)       header.current_validation_step;
  m_current_testing_step = (int)          header.current_testing_step;
  m_max_mini_batch_size = (int)           header.max_mini_batch_size;
  m_current_mini_batch_size = (int)       header.current_mini_batch_size;
  m_current_phase      =                  header.current_phase;

  p.set_cb_type((callback_type) header.callback_type);
  load_rng_from_checkpoint_shared(p, m_comm);

  for (weights *w : m_weights) {
    w->load_from_checkpoint_distributed(p);
  }

  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->load_from_checkpoint_distributed(p)) {
      return false;
    }
  }
  if(get_num_iterations_per_epoch(execution_mode::validation) != 0){
    for (const auto& m : m_metrics) {
      m->load_from_checkpoint_distributed(p);
    }
  }
  return true;
}

void trainer::write_proto(lbann_data::Trainer* proto) {
  proto->Clear();
  if (m_comm->am_world_master())
    proto->set_mini_batch_size(m_max_mini_batch_size);
}


bool trainer::save_trainer() {
  for (auto* c : m_callbacks) {
    auto *cb = dynamic_cast<lbann_callback_save_trainer*>(c);
    if(cb != nullptr) {
      return cb->save_trainer(this);
    }
  }
  if(m_comm->am_trainer_master()) {
    LBANN_WARNING("save_trainer was called, but the callback_save_trainer was not loaded");
  }
  return false;
}
#endif

}  // namespace lbann
