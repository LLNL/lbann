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
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_debug.hpp"
#include "lbann/comm.hpp"

namespace lbann {

namespace {

/** Get human-readable string describing process rank. */
std::string rank_string(const lbann_comm& comm) {
  std::stringstream msg;
  msg << "rank " << comm.get_rank_in_world();
  if (comm.get_num_trainers() > 1) {
    msg << " (rank " << comm.get_rank_in_trainer()
        << " of model " << comm.get_trainer_rank() << ")";
  }
  return msg.str();
}

/** Get human-readable string describing layer. */
std::string layer_string(const Layer& l) {
  return l.get_type() + " layer \"" + l.get_name() + "\"";
}

/** Get human-readable string describing weights and optimizer. */
std::string weights_string(const weights& w) {
  std::stringstream msg;
  msg << "weights \"" << w.get_name() << "\" (";
  const auto* opt = w.get_optimizer();
  if (opt == nullptr) { msg << "no"; }
  else { msg << opt->get_type(); }
  msg << " optimizer)";
  return msg.str();
}

/** Get human-readable string describing current batch step. */
std::string batch_step_string(const model& m) {
  std::stringstream msg;
  const auto& mode = m.get_execution_mode();
  msg << _to_string(mode) << " batch " << m.get_step();
  msg << " (epoch " << m.get_epoch() << ")";
  return msg.str();
}

} // namespace

// Status updates for batch beginnings/endings
void lbann_callback_debug::on_batch_begin(model *m) {
  if(m_modes.empty() || m_modes.count(m->get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": "
        << "starting " << batch_step_string(*m) << std::endl;
    std::cerr << msg.str();
  }
}
void lbann_callback_debug::on_batch_end(model *m) {
  if(m_modes.empty() || m_modes.count(m->get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": "
        << "ending " << batch_step_string(*m) << std::endl;
    std::cerr << msg.str();
  }
}
void lbann_callback_debug::on_batch_evaluate_begin(model *m) {
  on_batch_begin(m);
}
void lbann_callback_debug::on_batch_evaluate_end(model *m) {
  on_batch_end(m);
}

// Status updates for beginning/ending of layer forward/backward prop
void lbann_callback_debug::on_forward_prop_begin(model *m, Layer *l) {
  if(m_modes.empty() || m_modes.count(m->get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": " << layer_string(*l)
        << " is starting forward prop for " << batch_step_string(*m)
        << std::endl;
    std::cerr << msg.str();
  }
}
void lbann_callback_debug::on_forward_prop_end(model *m, Layer *l) {
  if(m_modes.empty() || m_modes.count(m->get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": " << layer_string(*l)
        << " is   ending forward prop for " << batch_step_string(*m)
        << std::endl;
    std::cerr << msg.str();
  }
}
void lbann_callback_debug::on_backward_prop_begin(model *m, Layer *l) {
  if(m_modes.empty() || m_modes.count(m->get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": " << layer_string(*l)
        << " is starting backward prop for " << batch_step_string(*m)
        << std::endl;
    std::cerr << msg.str();
  }
}
void lbann_callback_debug::on_backward_prop_end(model *m, Layer *l) {
  if(m_modes.empty() || m_modes.count(m->get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": " << layer_string(*l)
        << " is   ending backward prop for " << batch_step_string(*m)
        << std::endl;
    std::cerr << msg.str();
  }
}
void lbann_callback_debug::on_evaluate_forward_prop_begin(model *m, Layer *l) {
  on_forward_prop_begin(m, l);
}
void lbann_callback_debug::on_evaluate_forward_prop_end(model *m, Layer *l) {
  on_backward_prop_end(m, l);
}

// Status updates for optimization step
void lbann_callback_debug::on_optimize_begin(model *m, weights *w) {
  std::stringstream msg;
  msg << rank_string(*m->get_comm()) << ": " << weights_string(*w)
      << " is starting optimization step for " << batch_step_string(*m)
      << std::endl;
  std::cerr << msg.str();
}
void lbann_callback_debug::on_optimize_end(model *m, weights *w) {
  std::stringstream msg;
  msg << rank_string(*m->get_comm()) << ": " << weights_string(*w)
      << " is   ending optimization step for " << batch_step_string(*m)
      << std::endl;
  std::cerr << msg.str();
}

} // namespace lbann
