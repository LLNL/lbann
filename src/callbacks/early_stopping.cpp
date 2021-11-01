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
//
// lbann_early_stopping .hpp .cpp - Callback hooks for early stopping
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/early_stopping.hpp"
#include "lbann/utils/serialize.hpp"

#include <callbacks.pb.h>

#include <iostream>

namespace lbann {
namespace callback {

early_stopping::early_stopping(int64_t patience) :
  callback_base(), m_patience(patience) {}

early_stopping::early_stopping() :
  early_stopping(0)
{}

template <class Archive>
void early_stopping::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_patience),
     CEREAL_NVP(m_last_score),
     CEREAL_NVP(m_wait));
}

/// Monitor the objective function to see if the validation score
/// continues to improve
void early_stopping::on_validation_end(model *m) {
  auto& c = dynamic_cast<SGDExecutionContext&>(m->get_execution_context());
  execution_mode mode = c.get_execution_mode();
  EvalType score = m->get_objective_function()->get_mean_value(mode);
  if (score < m_last_score) {
    if (m->get_comm()->am_trainer_master()) {
      std::cout << "Model " << m->get_comm()->get_trainer_rank() <<
        " early stopping: score is improving " << m_last_score << " >> " <<
        score << std::endl;
    }
    m_last_score = score;
    m_wait = 0;
  } else {
    if (m_wait >= m_patience) {
      c.set_early_stop(true);
      if (m->get_comm()->am_trainer_master()) {
        std::cout << "Model " << m->get_comm()->get_trainer_rank() <<
          " terminating training due to early stopping: " << score <<
          " score and " << m_last_score << " last score" << std::endl;
      }
    } else {
      ++m_wait;
    }
  }
}

std::unique_ptr<callback_base>
build_early_stopping_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackEarlyStopping&>(proto_msg);
  return make_unique<early_stopping>(params.patience());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::early_stopping
#include <lbann/macros/register_class_with_cereal.hpp>
