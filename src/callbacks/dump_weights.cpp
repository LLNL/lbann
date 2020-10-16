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
// dump_weights .hpp .cpp - Callbacks to dump weight matrices
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/dump_weights.hpp"
#include "lbann/callbacks/checkpoint.hpp" // Reuse the checkpoint naming scheme
#include "lbann/utils/memory.hpp"
#include "lbann/weights/data_type_weights.hpp"
//#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/trainer_file_utils.hpp"

#include <callbacks.pb.h>

#include <string>

namespace lbann {
namespace callback {

void dump_weights::on_train_begin(model *m) {
  do_dump_weights(*m, "initial");
}

void dump_weights::on_epoch_end(model *m) {
  do_dump_weights(*m);
}

void dump_weights::do_dump_weights(const model& m, std::string s) {
  const auto& c = static_cast<const sgd_execution_context&>(m.get_execution_context());

  if(c.get_epoch() % m_epoch_interval != 0)  return;

  // Create directory
  auto& t = c.get_trainer();
  std::string epochdir = El::BuildString(get_shared_checkpoint_dirname(t.get_name(),
                                                                       c.get_training_algorithm().get_name(),
                                                                       m_directory.c_str(),
                                                                       c.get_execution_mode(),
                                                                       c.get_epoch(),
                                                                       c.get_step()),

                                         m.get_name(), '/');
  file::trainer_master_make_directory(epochdir, m.get_comm());
  for (weights *w : m.get_weights()) {
    // create weight file name to match to weight list entry
    const auto* dtw = dynamic_cast<const data_type_weights<DataType>*>(w);
    auto file = El::BuildString(epochdir, "model_weights_", w->get_name(), "_",
                                dtw->get_values().Height(), "x",
                                dtw->get_values().Width());

    El::Write(dtw->get_values(), file, El::ASCII);
  }
  if (m.get_comm()->am_trainer_master()) {
    auto latest_file = get_last_shared_checkpoint_filename(t.get_name(),
                                                           c.get_training_algorithm().get_name(),
                                                           m_directory.c_str());
    write_latest(latest_file, c.get_execution_mode(), c.get_epoch(), c.get_step());
  }

}

std::unique_ptr<callback_base>
build_dump_weights_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDumpWeights&>(proto_msg);
  return make_unique<dump_weights>(params.directory(), params.epoch_interval());
}

} // namespace callback
} // namespace lbann
