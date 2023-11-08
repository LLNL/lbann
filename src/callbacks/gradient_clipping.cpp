////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
// gradient_clipping .hpp .cpp - Callbacks to clip gradient values in training
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/gradient_clipping.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/loss/l2_norm2.hpp"
#include "lbann/models/model.hpp"
#include "lbann/optimizers/data_type_optimizer.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/weights.hpp"

#include "callback_helpers.hpp"
#include "lbann/proto/callbacks.pb.h"

#include <vector>

namespace lbann {
namespace callback {

clip_gradient_norm::clip_gradient_norm()
  : clip_gradient_norm(std::vector<std::string>{})
{}

void clip_gradient_norm::setup(model* m)
{

  // Add all weights if list of weights is not initialized
  std::vector<weights*> weights_list =
    select_things_by_name(m->get_weights(), m_weight_names);
  if (weights_list.empty()) {
    weights_list = m->get_weights();
  }

  // Remove weights that are not being optimized
  std::unordered_set<weights*>().swap(m_weights);
  for (weights* w : weights_list) {
    if (w->has_optimizer()) {
      m_weights.insert(w);
    }
  }
}

template <class Archive>
void clip_gradient_norm::serialize(Archive& ar)
{
  ar(cereal::base_class<callback_base>(this),
     CEREAL_NVP(m_weight_names),
     CEREAL_NVP(m_global_norm),
     CEREAL_NVP(m_value));
}

void clip_gradient_norm::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_clip_gradient_norm();
  msg->set_weights(protobuf::to_space_sep_string(this->m_weight_names));
  msg->set_global_norm(m_global_norm);
  msg->set_value(m_value);
}

void clip_gradient_norm::on_backward_prop_end(model* m)
{
  DataType global_norm = 0;
  for (weights* w : this->m_weights) {
    optimizer* opt = w->get_optimizer();
    if (opt != nullptr) {
      DataType norm;
      auto* dt_opt = dynamic_cast<data_type_optimizer<DataType>*>(opt);
      auto& grad = dt_opt->get_gradient_sharded();
      norm = El::Nrm2(grad);

      if (!m_global_norm && norm > this->m_value) {
        El::Scale(this->m_value / norm, grad);
      }
      else if (m_global_norm) {
        global_norm += norm * norm;
      }
    }
  }

  if (m_global_norm) {
    global_norm = std::sqrt(global_norm);
    if (global_norm > this->m_value) {
      DataType scale = this->m_value / global_norm;
      for (weights* w : this->m_weights) {
        optimizer* opt = w->get_optimizer();
        if (opt != nullptr) {
          auto* dt_opt = dynamic_cast<data_type_optimizer<DataType>*>(opt);
          auto& grad = dt_opt->get_gradient_sharded();
          El::Scale(scale, grad);
        }
      }
    }
  }
}

std::unique_ptr<callback_base> build_clip_gradient_norm_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackClipGradientNorm&>(
      proto_msg);
  return std::make_unique<clip_gradient_norm>(
    parse_list<std::string>(params.weights()),
    params.global_norm(),
    params.value());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::clip_gradient_norm
#define LBANN_CLASS_LIBNAME callback_clip_gradient_norm
#include <lbann/macros/register_class_with_cereal.hpp>
