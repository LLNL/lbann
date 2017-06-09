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
// lbann_model_dnn .hpp .cpp - Deep Neural Networks models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_GREEDY_LAYERWISE_AUTOENCODER_HPP
#define LBANN_MODEL_GREEDY_LAYERWISE_AUTOENCODER_HPP

#include "lbann/models/lbann_model_sequential.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include <vector>
#include <string>

namespace lbann {
class greedy_layerwise_autoencoder : public sequential_model {
 public:
  /// Constructor
  greedy_layerwise_autoencoder(uint mini_batch_size,
                               lbann_comm *comm,
                               objective_functions::objective_fn *obj_fn,
                               layer_factory *_layer_fac,
                               optimizer_factory *_optimizer_fac);

  /// Destructor
  ~greedy_layerwise_autoencoder();

  /// Save model to shared checkpoint
  bool save_to_checkpoint_shared(persist& p);

  /// Restore model from shared checkpoint
  bool load_from_checkpoint_shared(persist& p);

  /// Compute layer summaries
  void summarize(lbann_summary& summarizer);

  /// Train neural network
  /** @param num_epochs Number of epochs to train at each phase
   *  @param evaluation_frequency How often to evaluate model on
   *  validation set. A value less than 1 will disable evaluation.
   */
  void train(int num_epochs, int evaluation_frequency=0);

  // Train each phase ( a set of (original) input, hidden and mirror layers (output))
  void train_phase(int num_epochs, int evaluation_frequency);

  /// Training step on one mini-batch
  bool train_mini_batch();

  ///Global evaluation (testing), provide overall cost relative to original input
  void evaluate(execution_mode mode=execution_mode::testing);
  /// Evaluate (validation) per phase
  void evaluate_phase(execution_mode mode=execution_mode::validation);
  /// Evaluation step on one mini-batch
  bool evaluate_mini_batch();

  const std::string& name() {
    return m_name;
  }
  void reset_phase();

 protected:
  /// Model's name
  std::string m_name;
  /// index of last layer in a phase
  size_t m_phase_end;
  /// containers for  mirror layers
  std::vector<Layer *> m_reconstruction_layers;

  /// Flag recording whether we have a mirror layer inserted in model for training
  uint32_t m_have_mirror;

  /// Inserts a mirror layer for specified layer index
  void insert_mirror(uint32_t layer_index);

  /// Removes mirror for specified layer index
  void remove_mirror(uint32_t layer_index);
};
}


#endif // LBANN_MODEL_DNN_HPP
