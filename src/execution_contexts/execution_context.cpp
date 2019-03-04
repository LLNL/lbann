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

#include "lbann/training_algorithms/training_algorithm.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/callbacks/callback.hpp"
//#include "lbann/callbacks/callback_save_training_algorithm.hpp"
#include "lbann/io/persist.hpp"
// #include "lbann/layers/io/input/generic_input_layer.hpp"
// #include "lbann/layers/transform/dummy.hpp"
// #include "lbann/layers/transform/split.hpp"
// #include "lbann/layers/transform/evaluation.hpp"
// #include "lbann/objective_functions/layer_term.hpp"
// #include "lbann/metrics/layer_metric.hpp"
// #include "lbann/utils/random.hpp"
// #include "lbann/utils/omp_diagnostics.hpp"
// #include "lbann/utils/description.hpp"
#include <string>
#include <unistd.h>
#include <iomanip>
#include <queue>
#include <unordered_set>
#include <lbann.pb.h>

#include "mpi.h"

namespace lbann {

//******************************************************************************
// Execution context
//******************************************************************************

execution_context::execution_context(observing_ptr<trainer> trainer, lbann_comm *comm, execution_mode mode)
  : m_trainer(trainer),
    m_comm(comm),
    m_execution_mode(mode),
    m_terminate_training(false) {}

////////////////////////////////////////////////////////////
// Training_Algorithm state
////////////////////////////////////////////////////////////

// observing_ptr<thread_pool> training_algorithm::get_io_thread_pool() {
//   return m_trainer->get_io_thread_pool();
// }

observing_ptr<thread_pool> execution_context::get_io_thread_pool() const {
  return m_trainer->get_io_thread_pool();
}

/** Are background I/O activities enabled by the input layers */
bool execution_context::background_io_activity_allowed() {
  return m_trainer->background_io_activity_allowed();
}


  #if 0
////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

/* struct used to serialize mode fields in file and MPI transfer */
struct lbann_execution_context_header {
  uint32_t execution_mode;
  uint32_t terminate_training;
  uint64_t current_epoch;
  uint64_t current_step;
  uint64_t current_validation_step;
  uint64_t current_testing_step;
  uint32_t callback_type;
};

bool execution_context::save_to_checkpoint_shared(persist& p) {
  // write out fields we need to save for execution_context
  if (p.get_cb_type() != callback_type::validation) {
    if (m_comm->am_trainer_master()) {
      p.write_uint32(persist_type::train, "execution_mode",     (uint32_t) m_execution_mode);
      p.write_uint32(persist_type::train, "terminate_training", (uint32_t) m_terminate_training);
      p.write_uint64(persist_type::train, "current_epoch",      (uint64_t) m_current_epoch);
      p.write_uint64(persist_type::train, "current_step",       (uint64_t) m_current_step);
      p.write_uint64(persist_type::train, "current_testing_step",       (uint64_t) m_current_testing_step);
      p.write_uint32(persist_type::train, "persist_callback_type",      (uint32_t) p.get_cb_type());
      if(p.get_cb_type() == callback_type::batch)
        p.write_uint64(persist_type::validate, "current_validataion_step",       (uint64_t) m_current_validation_step);
    }

    if(p.get_cb_type() == callback_type::batch || get_num_iterations_per_epoch(execution_mode::validation) == 0){
      save_rng_to_checkpoint_shared(p, m_comm);
      for (const auto& m : m_metrics) {
        m->save_to_checkpoint_shared(p);
      }
    }
  }
  else{
    if (m_comm->am_trainer_master()) {
      p.write_uint64(persist_type::validate, "current_validataion_step",       (uint64_t) m_current_validation_step);
    }
    save_rng_to_checkpoint_shared(p, m_comm);
  }
  return true;
}

bool execution_context::load_from_checkpoint_shared(persist& p) {
  // have rank 0 read the file
  // read state from file
  struct lbann_execution_context_header header;
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

bool execution_context::save_to_checkpoint_distributed(persist& p){
  // write out fields we need to save for execution_context
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

bool execution_context::load_from_checkpoint_distributed(persist& p){
  struct lbann_execution_context_header header;
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
#endif

}  // namespace lbann
