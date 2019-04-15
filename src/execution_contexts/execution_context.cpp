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
#include "lbann/io/persist.hpp"
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

////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

/* struct used to serialize mode fields in file and MPI transfer */
struct lbann_execution_context_header {
  uint32_t execution_mode;
  uint32_t terminate_training;
  uint64_t current_step;
};

bool execution_context::save_to_checkpoint_shared(persist& p) {
  // write out fields we need to save for execution_context
  const persist_type pt = execution_mode_to_persist_type(m_execution_mode);

  // if (p.get_cb_type() != callback_type::validation) {
  if (m_comm->am_trainer_master()) {
    p.write_uint32(pt, "execution_mode",     (uint32_t) m_execution_mode);
    p.write_uint32(pt, "terminate_training", (uint32_t) m_terminate_training);
    p.write_uint64(pt, "current_step",       (uint64_t) m_step);
  }
      // if(p.get_cb_type() == callback_type::batch)
      //   p.write_uint64(persist_type::validate, "current_validataion_step",       (uint64_t) m_current_validation_step);
  // }
  // else{
  //   if (m_comm->am_trainer_master()) {
  //     p.write_uint64(persist_type::validate, "current_validataion_step",       (uint64_t) m_current_validation_step);
  //   }
  // }
  return true;
}

bool execution_context::load_from_checkpoint_shared(persist& p) {
  // have rank 0 read the file
  // read state from file
  struct lbann_execution_context_header header;

  callback_type cb = p.get_cb_type();
  const persist_type pt = callback_type_to_persist_type(cb);

  // Assume checkpoint reload from epoch end not step end
  if (m_comm->am_trainer_master()) {
      p.read_uint32(pt, "execution_mode",     &header.execution_mode);
      p.read_uint32(pt, "terminate_training", &header.terminate_training);
      p.read_uint64(pt, "current_step",       &header.current_step);
  }

  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  m_comm->trainer_broadcast(0, header);
  // set our member params from values read from disk
  m_execution_mode     = (execution_mode) header.execution_mode;
  m_terminate_training = (bool)           header.terminate_training;
  m_step               = (El::Int)        header.current_step;

  return true;
}

bool execution_context::save_to_checkpoint_distributed(persist& p){
  // write out fields we need to save for execution_context
  const persist_type pt = execution_mode_to_persist_type(m_execution_mode);

  p.write_uint32(pt, "execution_mode",     (uint32_t) m_execution_mode);
  p.write_uint32(pt, "terminate_training", (uint32_t) m_terminate_training);
  p.write_uint64(pt, "current_step",       (uint64_t) m_step);

  return true;
}

bool execution_context::load_from_checkpoint_distributed(persist& p){
  struct lbann_execution_context_header header;
  callback_type cb = p.get_cb_type();
  const persist_type pt = callback_type_to_persist_type(cb);

  // Assume checkpoint reload from epoch end not step end
  p.read_uint32(pt, "execution_mode",     &header.execution_mode);
  p.read_uint32(pt, "terminate_training", &header.terminate_training);
  p.read_uint64(pt, "current_step",       &header.current_step);

  // set our member params from values read from disk
  m_execution_mode     = (execution_mode) header.execution_mode;
  m_terminate_training = (bool)           header.terminate_training;
  m_step               = (El::Int)        header.current_step;

  return true;
}

}  // namespace lbann
