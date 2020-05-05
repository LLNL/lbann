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
// set_weights_value .hpp .cpp - Callbacks to dump weight matrices
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/set_weights_value.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include <callbacks.pb.h>


namespace lbann {
namespace callback {

void set_weights_value::on_epoch_begin(model *m) {
  const auto& c = static_cast<const sgd_execution_context&>(m->get_execution_context());
  if(El::Int(c.get_epoch()) !=  m_epoch_interval)  return;
  for (weights *w : m->get_weights()) {
    auto* dtw = dynamic_cast<data_type_weights<DataType>*>(w);
    if(dtw->get_name() == m_weight_name) {
      dtw->set_value(DataType(m_weight_value),0);
    }
  }    
}


std::unique_ptr<callback_base>
build_set_weights_value_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSetWeightsValue&>(proto_msg);
  return make_unique<set_weights_value>(params.weight_name(), params.weight_value(), params.epoch_interval());
}

} // namespace callback
} // namespace lbann
