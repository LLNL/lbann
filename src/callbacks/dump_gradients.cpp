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
// dump_gradients .hpp .cpp - Callbacks to dump gradients
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/dump_gradients.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/optimizers/data_type_optimizer.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/weights.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <vector>

namespace lbann {
namespace callback {

dump_gradients::dump_gradients() : dump_gradients("", 0) {}

template <class Archive>
void dump_gradients::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_basename));
}

void dump_gradients::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_dump_gradients();
  msg->set_basename(m_basename);
  msg->set_interval(m_batch_interval);
}

void dump_gradients::on_backward_prop_end(model* m)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  for (weights* w : m->get_weights()) {
    optimizer* opt = w->get_optimizer();
    if (opt != nullptr) {
      const std::string file =
        (m_basename + "model" +
         std::to_string(m->get_comm()->get_trainer_rank()) + "-epoch" +
         std::to_string(c.get_epoch()) + "-step" +
         std::to_string(c.get_step()) + "-" + w->get_name() + "-Gradient");
      auto* dt_opt = dynamic_cast<data_type_optimizer<DataType>*>(opt);
      auto& grad = dt_opt->get_gradient_sharded();
      if (grad.Participating()) {
        El::Write(grad, file, El::ASCII);
      }
    }
  }
}

std::unique_ptr<callback_base> build_dump_gradients_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDumpGradients&>(proto_msg);
  return std::make_unique<dump_gradients>(params.basename(), params.interval());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::dump_gradients
#define LBANN_CLASS_LIBNAME callback_dump_gradients
#include <lbann/macros/register_class_with_cereal.hpp>
