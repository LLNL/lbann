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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_replace_weights.hpp"
#include "lbann/proto/proto_common.hpp"

#include "callback_helpers.hpp"

namespace lbann {

void lbann_callback_replace_weights::setup(model *m) {
  auto const layers = m->get_layers();
  m_src_layers = select_things_by_name(layers, m_src_layer_names);
  m_dst_layers = select_things_by_name(layers, m_dst_layer_names);

  // Pretend the extra storage space matters
  std::vector<std::string>().swap(m_src_layer_names);
  std::vector<std::string>().swap(m_dst_layer_names);
}

void lbann_callback_replace_weights::on_batch_end(model *m) {
  const auto& step = m->get_step(execution_mode::training);
  if(step % m_batch_interval == 0) {
    for(size_t i = 0; i < m_src_layers.size(); i++) {
      m_dst_layers[i]->replace_weights(m_src_layers[i]);
    }
  }
}

std::unique_ptr<lbann_callback>
build_callback_replace_weights_from_pbuf(
  const google::protobuf::Message& proto_msg, lbann_summary*) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackReplaceWeights&>(proto_msg);
  return make_unique<lbann_callback_replace_weights>(
    parse_list<std::string>(params.source_layers()),
    parse_list<std::string>(params.destination_layers()),
    params.batch_interval());
}

}  // namespace lbann
