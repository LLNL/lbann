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

#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

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

bool execution_context::save_to_checkpoint_shared(persist& p) {
  if (get_comm()->am_trainer_master()) {
    write_cereal_archive<execution_context>(*this, p, get_execution_mode(), "_ctx.xml");
  }

  return true;
}

bool execution_context::load_from_checkpoint_shared(persist& p) {
  bool success = false;
  std::string buf;

  // Assume checkpoint reload from epoch end not step end
  if (get_comm()->am_trainer_master()) {
    success = read_cereal_archive<execution_context>(*this, p, get_execution_mode(), "_ctx.xml");
    buf = create_cereal_archive_binary_string<execution_context>(*this);
  }

  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  get_comm()->trainer_broadcast(0, buf);

  if (!get_comm()->am_trainer_master()) {
    success = unpack_cereal_archive_binary_string<execution_context>(*this, buf);
  }

  return success;
}

bool execution_context::save_to_checkpoint_distributed(persist& p){
  return write_cereal_archive<execution_context>(*this, p, get_execution_mode(), "_ctx.xml");
}

bool execution_context::load_from_checkpoint_distributed(persist& p){
  return read_cereal_archive<execution_context>(*this, p, get_execution_mode(), "_ctx.xml");
}

}  // namespace lbann
