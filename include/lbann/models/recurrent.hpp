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
// recurrent .hpp .cpp - Recurrent neural network models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_RECURRENT_HPP
#define LBANN_MODEL_RECURRENT_HPP

#include "lbann/models/directed_acyclic_graph.hpp"

namespace lbann {

/** Recurrent neural network model. 
 *  Training is performed with back propagation through time, i.e. by
 *  unrolling the recurrent network into a DAG.
 */
class recurrent_model : public directed_acyclic_graph_model {
 public:

  /** Constructor. */
  recurrent_model(lbann_comm *comm,
                int mini_batch_size,
                objective_function *obj_fn,
                optimizer* default_optimizer,
                int unroll_depth);

  /** Copy constructor. */
  recurrent_model(const recurrent_model& other) = default;
  /** Copy assignment operator. */
  recurrent_model& operator=(const recurrent_model& other) = default;
  /** Destructor. */
  ~recurrent_model() override = default;

  /** Create copy. */
  recurrent_model* copy() const override { return new recurrent_model(*this); }

  /** Get model name. */
  std::string name() const override { return "recurrent_model"; }

 protected:

  void setup_layer_topology() override;
  void setup_layers() override;

 private:

  /** The number of times to unroll the recurrent network. */
  int m_unroll_depth;

  /** Map to corresponding layer in previous step.
   *  Layers in the first step map to null pointers. The input and
   *  target layers map to themselves.
   */
  std::unordered_map<const Layer*, Layer*> m_previous_step_layer;
  /** Map to corresponding layer in next step.
   *  Layers in the last step map to null pointers. The input and
   *  target layers map to themselves.
   */
  std::unordered_map<const Layer*, Layer*> m_next_step_layer;

};

}  // namespace lbann

#endif  // LBANN_MODEL_RECURRENT_HPP
