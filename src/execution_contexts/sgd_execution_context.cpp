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
#include "lbann/base.hpp"
#include "lbann/io/persist_impl.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/serialize.hpp"

namespace lbann {

sgd_execution_context::sgd_execution_context(execution_mode mode,
                                             size_t mini_batch_size)
  : m_current_mini_batch_size(mini_batch_size),
    m_effective_mini_batch_size(mini_batch_size), m_execution_mode(mode)
{}

template <class Archive> void sgd_execution_context::serialize(Archive& ar)
{
  ar(cereal::base_class<execution_context>(this),
     CEREAL_NVP(m_epoch),
     CEREAL_NVP(m_current_mini_batch_size),
     CEREAL_NVP(m_effective_mini_batch_size),
     CEREAL_NVP(m_execution_mode));
}

////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

void sgd_execution_context::save_to_checkpoint_shared(persist& p)
{
  if (get_trainer().get_comm()->am_trainer_master()) {
    write_cereal_archive<sgd_execution_context>(*this,
                                                p,
                                                get_execution_mode(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                                "_ctx.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                                "_ctx.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
    );
  }
  return;
}

void sgd_execution_context::load_from_checkpoint_shared(persist& p)
{
  load_from_shared_cereal_archive<sgd_execution_context>(
    *this,
    p,
    get_execution_mode(),
    *(get_trainer().get_comm()),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
    "_ctx.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
    "_ctx.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return;
}

void sgd_execution_context::save_to_checkpoint_distributed(persist& p)
{
  write_cereal_archive<sgd_execution_context>(*this,
                                              p,
                                              get_execution_mode(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                              "_ctx.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                              "_ctx.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return;
}

void sgd_execution_context::load_from_checkpoint_distributed(persist& p)
{
  read_cereal_archive<sgd_execution_context>(*this,
                                             p,
                                             get_execution_mode(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                             "_ctx.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                             "_ctx.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return;
}

std::string sgd_execution_context::get_type() const { return "sgd"; }

bool seconds_termination_criteria::is_done(
  sgd_execution_context const& c) const noexcept
{
  auto const& comm = *(get_const_trainer().get_comm());
  int stop = (comm.am_trainer_master() &&
              (c.get_current_execution_time() >= m_max_seconds));
  if (comm.get_procs_per_trainer() > 1)
    comm.trainer_broadcast(0, stop);
  return (stop == 1);
}

} // namespace lbann

#define LBANN_CLASS_NAME sgd_execution_context
#include <lbann/macros/register_class_with_cereal.hpp>
