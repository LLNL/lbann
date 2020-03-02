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
// monitor_io .hpp .cpp - Callback hooks for I/O monitoring
////////////////////////////////////////////////////////////////////////////////

#include <utility>

#include "lbann/callbacks/monitor_io.hpp"
#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/proto/proto_common.hpp"

#include <callbacks.pb.h>

namespace lbann {
namespace callback {

void monitor_io::on_epoch_end(model *m) {
  const auto& c = static_cast<const sgd_execution_context&>(m->get_execution_context());
  lbann_comm *comm = m->get_comm();
  for (Layer *layer : m->get_layers()) {
    if(m_layers.size() == 0
       || m_layers.find(layer->get_name()) != m_layers.end()) {
      auto *input = dynamic_cast<generic_input_layer<DataType> *> (layer);
      if(input != nullptr) {
        std::cout << "Rank " << comm->get_trainer_rank() << "."
                  << comm->get_rank_in_trainer() << " processed "
                  << input->get_num_samples_trained() << " training samples of "
                  << input->get_total_num_training_samples() << " ("
                  << input->get_num_samples_trained() / c.get_epoch() << " per epoch)" << std::endl;
      }
    }
  }
}

void monitor_io::on_test_end(model *m) {
  const auto& c = static_cast<const sgd_execution_context&>(m->get_execution_context());
  lbann_comm *comm = m->get_comm();
  for (Layer *layer : m->get_layers()) {
    if(m_layers.size() == 0
       || m_layers.find(layer->get_name()) != m_layers.end()) {
      auto *input = dynamic_cast<generic_input_layer<DataType> *> (layer);
      if(input != nullptr) {
        std::cout << "Rank " << comm->get_trainer_rank() << "."
                  << comm->get_rank_in_trainer() << " processed "
                  << input->get_num_samples_tested() << " test samples of "
                  << input->get_total_num_testing_samples() << " ("
                  << input->get_num_samples_tested() / c.get_epoch()
                  << " per epoch)" << std::endl;
      }
    }
  }
}

std::unique_ptr<callback_base>
build_monitor_io_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDispIOStats&>(proto_msg);
  return make_unique<monitor_io>(
    parse_list<std::string>(params.layers()));
}

} // namespace callback
} // namespace lbann
