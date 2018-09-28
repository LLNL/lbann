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

#ifndef LBANN_MODEL_DIRECTED_ACYCLIC_GRAPH_HPP
#define LBANN_MODEL_DIRECTED_ACYCLIC_GRAPH_HPP

#include "lbann/models/model.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

/** Directed acyclic graph neural network model. */
class directed_acyclic_graph_model : public model {
 public:

  /** Constructor. */
  directed_acyclic_graph_model(lbann_comm *comm,
                               int max_mini_batch_size,
                               objective_function *obj_fn,
                               optimizer *default_optimizer);

  /** Copy constructor. */
  directed_acyclic_graph_model(const directed_acyclic_graph_model& other) = default;

  /** Copy assignment operator. */
  directed_acyclic_graph_model& operator=(const directed_acyclic_graph_model& other) = default;

  /** Destructor. */
  ~directed_acyclic_graph_model() override = default;

  /** Create copy. */
  directed_acyclic_graph_model* copy() const override { return new directed_acyclic_graph_model(*this); }

  /** Get model name. */
  std::string type() const override { return "directed_acyclic_graph_model"; }

 protected:

  /** For general DAG models, users need to manually specify each layer to
   *  freeze in the model description prototext.
   */
  void freeze_layers_under_frozen_surface() override {}

  /** Set up layer execution order.
   *  Called in setup function. A topological sort applied is to the
   *  layer list so that we can traverse a directed acyclic graph
   *  without violating dependencies.
   */
  void setup_layer_execution_order() override;

};

}  // namespace lbann

#endif  // LBANN_MODEL_DIRECTED_ACYCLIC_GRAPH_HPP
