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
//
// directed_acyclic_graph .hpp .cpp - Directed acyclic graph neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/directed_acyclic_graph.hpp"
#include <stack>
#include <unordered_map>

namespace lbann {

directed_acyclic_graph_model::directed_acyclic_graph_model(lbann_comm *comm,
                                                           int mini_batch_size,
                                                           objective_function *obj_fn,
                                                           optimizer* default_optimizer)
  : model(comm, mini_batch_size, obj_fn, default_optimizer) {}

void directed_acyclic_graph_model::setup_layer_execution_order() {
  model::setup_layer_execution_order();
  const auto& layer_graph = construct_layer_graph();
  const auto& sorted_order = graph::topological_sort(layer_graph);
  permute_layers(sorted_order);
}

}  // namespace lbann
