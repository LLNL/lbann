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

#include "lbann/execution_contexts/sgd_execution_context.hpp"

namespace lbann {

sgd_execution_context::sgd_execution_context(observing_ptr<trainer> trainer, lbann_comm *comm,
                                             execution_mode mode,
                                             int mini_batch_size)
  : execution_context(trainer, comm, mode),
    m_current_mini_batch_size(mini_batch_size),
    m_effective_mini_batch_size(mini_batch_size) {}

////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

/* struct used to serialize mode fields in file and MPI transfer */
struct lbann_execution_context_header {
  uint64_t current_epoch;
  uint64_t current_mini_batch_size;
  uint64_t effective_mini_batch_size;
};

bool sgd_execution_context::save_to_checkpoint_shared(persist& p) {
  // write out fields we need to save for execution_context
  const persist_type pt = execution_mode_to_persist_type(m_execution_mode);

  if (m_comm->am_trainer_master()) {
    p.write_uint64(pt, "current_epoch",             (uint64_t) m_epoch);
    p.write_uint64(pt, "current_mini_batch_size",   (uint64_t) m_current_mini_batch_size);
    p.write_uint64(pt, "effective_mini_batch_size", (uint64_t) m_effective_mini_batch_size);
  }

  return true;
}

bool sgd_execution_context::load_from_checkpoint_shared(persist& p) {
  // have rank 0 read the file
  // read state from file
  struct lbann_execution_context_header header;

  callback_type cb = p.get_cb_type();
  const persist_type pt = callback_type_to_persist_type(cb);

  // Assume checkpoint reload from epoch end not step end
  if (m_comm->am_trainer_master()) {
    p.read_uint64(pt, "current_epoch",     &header.current_epoch);
    p.read_uint64(pt, "current_mini_batch_size", &header.current_mini_batch_size);
    p.read_uint64(pt, "effective_mini_batch_size", &header.effective_mini_batch_size);
  }

  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  m_comm->trainer_broadcast(0, header);
  // set our member params from values read from disk
  m_epoch                     = (El::Int) header.current_epoch;
  m_current_mini_batch_size   = (El::Int) header.current_mini_batch_size;
  m_effective_mini_batch_size = (El::Int) header.effective_mini_batch_size;

  return true;
}

bool sgd_execution_context::save_to_checkpoint_distributed(persist& p) {
  // write out fields we need to save for execution_context
  const persist_type pt = execution_mode_to_persist_type(m_execution_mode);

  p.write_uint64(pt, "current_epoch",             (uint64_t) m_epoch);
  p.write_uint64(pt, "current_mini_batch_size",   (uint64_t) m_current_mini_batch_size);
  p.write_uint64(pt, "effective_mini_batch_size", (uint64_t) m_effective_mini_batch_size);

  return true;
}

bool sgd_execution_context::load_from_checkpoint_distributed(persist& p) {
  struct lbann_execution_context_header header;

  callback_type cb = p.get_cb_type();
  const persist_type pt = callback_type_to_persist_type(cb);

  // Assume checkpoint reload from epoch end not step end
  p.read_uint64(pt, "current_epoch",     &header.current_epoch);
  p.read_uint64(pt, "current_mini_batch_size", &header.current_mini_batch_size);
  p.read_uint64(pt, "effective_mini_batch_size", &header.effective_mini_batch_size);

  // set our member params from values read from disk
  m_epoch                     = (El::Int) header.current_epoch;
  m_current_mini_batch_size   = (El::Int) header.current_mini_batch_size;
  m_effective_mini_batch_size = (El::Int) header.effective_mini_batch_size;

  return true;
}

}  // namespace lbann
