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
// lbann_model_planar .hpp .cpp - Planar neural network models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_PLANAR_HPP
#define LBANN_MODEL_PLANAR_HPP

#include "lbann/models/model.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/layers/learning/learning.hpp"
#include <vector>
#include <string>

namespace lbann {

class planar_model : public model {
 public:

  /// Constructor
  planar_model(int mini_batch_size,
                   lbann_comm *comm,
                   objective_functions::objective_function *obj_fn,
                   optimizer_factory *optimizer_fac);
/**
  planar_model(const planar_model& other);
  planar_model& operator=(const planar_model& other); 
*/

  /// Destructor
  ~planar_model();

  /** Following functions are used to add a set of layers at given horizontal level
   *  on a planar space. The layers are added either by duplicating a single layer
   *  or placing individual layers. */
  virtual int stackup_tail(int hindex, Layer *new_layer);
  virtual int stackup_duplicate(Layer *new_layer);

  /// Setup planar model
  virtual void setup() override;
  virtual void setup_subset(int start_index, int end_index);

  /// Train model
  void train(int num_epochs) override;

  /// Training step on one mini-batch
  bool train_mini_batch() override;

  /// Evaluate model
  virtual void evaluate(execution_mode mode) override;

  /// Evaluation step on one mini-batch
  virtual bool evaluate_mini_batch() override;

  /** Return true if about to start a new training epoch */
  virtual bool at_epoch_start();

  /// Ensure weight matriecs in heads at each level are the same
  virtual void equalize(int  start_index, int end_index) = 0;
  /// Add weight matrices in heads at each level
  virtual void sum_up_gradients() = 0;

  /// Check if the model has a valid data set for the execution mode
  bool is_execution_mode_valid(execution_mode mode) override;

  virtual std::string name() const override { return "planar_model"; }

 protected:
  /// the maximum number of horizontal layers in the network
  int m_width;

  /// List of layers on the plane
  /// m_layers contains a set of horizontal layers for each level
  /// m_head_counts contains the number of horizontal layers for each level
  /// For now, the entries in m_head_counts are either 1 or m_width (to support
  /// the Siamese network)
  std::vector<std::vector<Layer*> > m_layers;
  std::vector<int> m_head_counts;
};

}  // namespace lbann

#endif  // LBANN_MODEL_PLANAR_HPP
