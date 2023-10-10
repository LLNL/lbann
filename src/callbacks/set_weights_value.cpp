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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/set_weights_value.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include "lbann/proto/callbacks.pb.h"

namespace lbann {
namespace callback {

set_weights_value::set_weights_value(std::string weights_name,
                                     double value,
                                     size_t step)
  : callback_base(),
    m_weights_name(std::move(weights_name)),
    m_value{value},
    m_step{step}
{}

set_weights_value::set_weights_value() : set_weights_value("", 0, 0) {}

template <class Archive>
void set_weights_value::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_weights_name),
     CEREAL_NVP(m_value),
     CEREAL_NVP(m_step));
}

void set_weights_value::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_set_weights_value();
  msg->set_weights(m_weights_name);
  msg->set_value(m_value);
  msg->set_step(m_step);
}

set_weights_value* set_weights_value::copy() const
{
  return new set_weights_value(*this);
}

std::string set_weights_value::name() const { return "set weights value"; }

void set_weights_value::on_batch_begin(model* m)
{

  // Check whether to set weights value at current mini-batch step
  const auto& context = m->get_execution_context();
  if (context.get_step() != m_step) {
    return;
  }
  if (context.get_execution_mode() != execution_mode::training) {
    return;
  }

  // Find weights and set value
  for (weights* w : m->get_weights()) {
    if (w->get_name() == m_weights_name) {
      /// @todo Handle weights with other data types
      auto* dtw = dynamic_cast<data_type_weights<float>*>(w);
      if (dtw == nullptr) {
        LBANN_ERROR("\"",
                    this->name(),
                    "\" callback ",
                    "attempted to set value of ",
                    "weights \"",
                    m_weights_name,
                    "\", "
                    "which has an invalid data type");
      }
      El::Fill(dtw->get_values_sharded(), float(m_value));
      return;
    }
    LBANN_ERROR("\"",
                this->name(),
                "\" callback ",
                "could not find ",
                "weights \"",
                m_weights_name,
                "\", "
                "in model \"",
                m->get_name(),
                "\"");
  }
}

std::unique_ptr<callback_base> build_set_weights_value_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSetWeightsValue&>(
      proto_msg);
  return std::make_unique<set_weights_value>(params.weights(),
                                             params.value(),
                                             params.step());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::set_weights_value
#define LBANN_CLASS_LIBNAME callback_set_weights_value
#include <lbann/macros/register_class_with_cereal.hpp>
