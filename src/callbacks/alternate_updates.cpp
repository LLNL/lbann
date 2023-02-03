////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#include "lbann/callbacks/alternate_updates.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/protobuf.hpp"

#include "callback_helpers.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <string>
#include <vector>

namespace lbann {
namespace callback {

void alternate_updates::setup(model *m) {
  auto const layers = m->get_layers();
  unfreeze_layers = select_things_by_name(layers, m_layer_names_1);
  freeze_layers = select_things_by_name(layers, m_layer_names_2);
}

void alternate_updates::on_batch_begin(model *m) {
  const auto& c = m->get_execution_context();
  const auto& step = c.get_step();

  if(step % m_iters_tot == 0ul || step % m_iters_tot == (unsigned long) m_iters_1) {
    for(size_t i = 0; i < freeze_layers.size(); i++){
      if(!freeze_layers[i])
        LBANN_ERROR("Layer pointer is null.");

      freeze_layers[i]->freeze();
    }

    for(size_t i = 0; i < unfreeze_layers.size(); i++){
      if(!unfreeze_layers[i])
        LBANN_ERROR("Layer pointer is null.");

      unfreeze_layers[i]->unfreeze();
    }

    freeze_layers.swap(unfreeze_layers);
  }
}

void alternate_updates::write_specific_proto(lbann_data::Callback& proto) const{
  auto* msg = proto.mutable_alternate_updates();
  msg->set_layers_1(protobuf::to_space_sep_string(m_layer_names_1));
  msg->set_layers_2(protobuf::to_space_sep_string(m_layer_names_2));
  msg->set_iters_1(m_iters_1);
  msg->set_iters_2(m_iters_tot - m_iters_1);
}

std::unique_ptr<callback_base>
build_alternate_updates_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackAlternateUpdates&>(proto_msg);
  return std::make_unique<alternate_updates>(
    parse_list<std::string>(params.layers_1()),
    parse_list<std::string>(params.layers_2()),
    params.iters_1(),
    params.iters_2());
}

} // namespace callback
} // namespace lbann
