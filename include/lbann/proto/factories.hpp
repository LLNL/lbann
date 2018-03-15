////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_PROTO_FACTORIES_HPP
#define LBANN_PROTO_FACTORIES_HPP

// Macro to throw an exception
#define LBANN_ERROR(comm, message)                              \
  do {                                                          \
    if (comm->am_world_master()) {                              \
      std::stringstream ss_LBANN_ERROR;                         \
      ss_LBANN_ERROR << "Error at "                             \
                     << __FILE__ << ":" << __LINE__ << ": "     \
                     << message;                                \
      throw lbann::lbann_exception(ss_LBANN_ERROR.str());       \
    }                                                           \
  } while(0)

#include "lbann/proto/proto_common.hpp"

namespace lbann {
namespace proto {

/** Parse a space-separated list. */
template <typename T = std::string>
std::vector<T> parse_list(std::string str) {
  std::vector<T> list;
  std::stringstream ss(str);
  for (T entry; ss >> entry;) {
    list.push_back(entry);
  }
  return list;
}

/** Construct a layer graph specified with a prototext. */
std::vector<Layer*> construct_layer_graph(lbann_comm* comm,
                                          std::map<execution_mode, generic_data_reader *>& data_readers,
                                          cudnn::cudnn_manager* cudnn,
                                          const lbann_data::Model& proto_model);

/** Construct a layer specified with prototext. */
template <data_layout layout>
Layer* construct_layer(lbann_comm* comm,
                       std::map<execution_mode, generic_data_reader*>& data_readers,
                       int num_parallel_readers,
                       cudnn::cudnn_manager* cudnn,
                       const lbann_data::Layer& proto_layer);

/** Construct a callback specified with prototext. */
lbann_callback* construct_callback(lbann_comm* comm,
                                   const lbann_data::Callback& proto_cb,
                                   std::map<execution_mode, generic_data_reader*>& data_readers,
                                   std::vector<Layer*> layer_list,
                                   std::vector<weights*> weights_list,
                                   lbann_summary* summarizer);

/** Construct a metric specified with prototext. */
metric* construct_metric(lbann_comm* comm,
                         const lbann_data::Metric& proto_metric);

/** Construct an optimizer specified with prototext. */
optimizer* construct_optimizer(lbann_comm* comm,
                               const lbann_data::Optimizer& proto_opt);

/** Construct an objective function specified with prototext. */
objective_function* construct_objective_function(const lbann_data::ObjectiveFunction& proto_obj);

} // namespace proto
} // namespace lbann

#endif // LBANN_PROTO_FACTORIES_HPP
