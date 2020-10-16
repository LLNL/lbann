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
// callback_sync_layers.cpp - Callback to synchronize layers
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/sync_layers.hpp"

#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/timer.hpp"

#include <callbacks.pb.h>

namespace lbann {
namespace callback {

void sync_layers::on_forward_prop_end(model *m, Layer *l) {
  if (m_only_input && dynamic_cast<generic_input_layer<DataType>*>(l) == nullptr) {
    return;  // Skip non-input layers.
  }
  double start = get_time();
  do_sync(l);
  l->m_fp_time += get_time() - start;
}

void sync_layers::on_backward_prop_end(model *m, Layer *l) {
  if (m_only_input) {
    return;
  }
  double start = get_time();
  do_sync(l);
  l->m_bp_time += get_time() - start;
}

void sync_layers::do_sync(Layer *l) {
#ifdef LBANN_HAS_GPU
  if (m_sync_gpus) {
    hydrogen::gpu::SynchronizeDevice();
  }
#endif
  if (m_sync_mpi) {
    l->get_comm()->global_barrier();
  }
}

std::unique_ptr<callback_base>
build_sync_layers_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSyncLayers&>(proto_msg);
  return make_unique<sync_layers>(params.sync_gpus(),
                                                 params.sync_mpi(),
                                                 params.only_input());
}

} // namespace callback
} // namespace lbann
