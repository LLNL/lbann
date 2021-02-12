////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/callbacks/debug.hpp"
#include "lbann/comm.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/weights/data_type_weights.hpp"
#include "lbann/utils/serialize.hpp"

#include "callbacks.pb.h"

namespace lbann {
namespace callback {

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
template <typename TensorDataType>
std::string weights_string(const data_type_weights<TensorDataType>& w) {
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
  const auto& c =
    dynamic_cast<const sgd_execution_context&>(m.get_execution_context());
  std::stringstream msg;
  const auto& mode = c.get_execution_mode();
  msg << to_string(mode) << " batch " << c.get_step();
  msg << " (epoch " << c.get_epoch() << ")";
  return msg.str();
}

} // namespace

template <class Archive>
void debug::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_modes));
}

// Status updates for batch beginnings/endings
void debug::on_batch_begin(model *m) {
  const auto& c = m->get_execution_context();
  if(m_modes.empty() || m_modes.count(c.get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": "
        << "starting " << batch_step_string(*m) << std::endl;
    std::cerr << msg.str();
  }
}
void debug::on_batch_end(model *m) {
  const auto& c = m->get_execution_context();
  if(m_modes.empty() || m_modes.count(c.get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": "
        << "ending " << batch_step_string(*m) << std::endl;
    std::cerr << msg.str();
  }
}
void debug::on_batch_evaluate_begin(model *m) {
  on_batch_begin(m);
}
void debug::on_batch_evaluate_end(model *m) {
  on_batch_end(m);
}

// Status updates for beginning/ending of layer forward/backward prop
void debug::on_forward_prop_begin(model *m, Layer *l) {
  const auto& c = m->get_execution_context();
  if(m_modes.empty() || m_modes.count(c.get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": " << layer_string(*l)
        << " is starting forward prop for " << batch_step_string(*m)
        << std::endl;
    std::cerr << msg.str();
  }
}
void debug::on_forward_prop_end(model *m, Layer *l) {
  const auto& c = m->get_execution_context();
  if(m_modes.empty() || m_modes.count(c.get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": " << layer_string(*l)
        << " is   ending forward prop for " << batch_step_string(*m)
        << std::endl;
    std::cerr << msg.str();
  }
}
void debug::on_backward_prop_begin(model *m, Layer *l) {
  const auto& c = m->get_execution_context();
  if(m_modes.empty() || m_modes.count(c.get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": " << layer_string(*l)
        << " is starting backward prop for " << batch_step_string(*m)
        << std::endl;
    std::cerr << msg.str();
  }
}
void debug::on_backward_prop_end(model *m, Layer *l) {
  const auto& c = m->get_execution_context();
  if(m_modes.empty() || m_modes.count(c.get_execution_mode()) > 0) {
    std::stringstream msg;
    msg << rank_string(*m->get_comm()) << ": " << layer_string(*l)
        << " is   ending backward prop for " << batch_step_string(*m)
        << std::endl;
    std::cerr << msg.str();
  }
}
void debug::on_evaluate_forward_prop_begin(model *m, Layer *l) {
  on_forward_prop_begin(m, l);
}
void debug::on_evaluate_forward_prop_end(model *m, Layer *l) {
  on_backward_prop_end(m, l);
}

// Status updates for optimization step
void debug::on_optimize_begin(model *m, weights *w) {
  auto& dtw = dynamic_cast<data_type_weights<DataType>&>(*w);
  std::stringstream msg;
  msg << rank_string(*m->get_comm()) << ": " << weights_string(dtw)
      << " is starting optimization step for " << batch_step_string(*m)
      << std::endl;
  std::cerr << msg.str();
}
void debug::on_optimize_end(model *m, weights *w) {
  auto& dtw = dynamic_cast<data_type_weights<DataType>&>(*w);
  std::stringstream msg;
  msg << rank_string(*m->get_comm()) << ": " << weights_string(dtw)
      << " is   ending optimization step for " << batch_step_string(*m)
      << std::endl;
  std::cerr << msg.str();
}

std::unique_ptr<callback_base>
build_debug_callback_from_pbuf(const google::protobuf::Message& proto_msg,
                               const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDebug&>(proto_msg);
  const auto& modes =
    parse_set<execution_mode>(params.phase());
  return make_unique<debug>(modes);
}

} // namespace callback
} // namespace lbann

// CEREAL_REGISTER_TYPE_WITH_NAME(
//   ::lbann::callback::debug,
//   "callback::debug")
