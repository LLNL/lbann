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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_GREEDY_LAYERWISE_AUTOENCODER_HPP
#define LBANN_MODEL_GREEDY_LAYERWISE_AUTOENCODER_HPP

#include "lbann/models/sequential.hpp"

namespace lbann {

/** Greedy layerwise autoencoder. */
class greedy_layerwise_autoencoder : public sequential_model {
 public:

  /** Constructor. */
  greedy_layerwise_autoencoder(lbann_comm *comm,
                               int mini_batch_size,
                               objective_function *obj_fn,
                               optimizer *default_optimizer);

  /** Copy constructor. */
  greedy_layerwise_autoencoder(const greedy_layerwise_autoencoder&);
  /** Copy assignment operator. */
  greedy_layerwise_autoencoder& operator=(const greedy_layerwise_autoencoder&);
  /** Destructor. */
  ~greedy_layerwise_autoencoder() override;
  /** Create copy. */
  greedy_layerwise_autoencoder* copy() const override {
    return new greedy_layerwise_autoencoder(*this);
  }

  /** Get model name. */
  std::string name() const override { return "greedy layerwise autoencoder"; }

  /** Train greedy layerwise autoencoder. */
  void train(int num_epochs, int num_batches=0) override;

 protected:

  /** Current training phase.
   *  If negative, the model is sequential.
   */
  int m_phase;
  /** Number of training phases. */
  int m_num_phases;
  /** Layer indices for the boundaries of model sections.
   *  The first half sections are considered as encoders and the
   *  second half as decoders. Each training phase consists of
   *  training an encoder and decoder while keeping the rest of the
   *  model frozen.
   */
  std::vector<int> m_sections;
  /** Reconstruction layer for training phases. */
  Layer* m_reconstruction;

  /** Set up topology of layer graph.
   *  Called in setup function. Determine model sections.
   */
  void setup_layer_topology() override;

  /** Set the greedy layerwise autoencoder to a training phase.
   *  During a phase, an encoder and decoder section of the model are
   *  trained while the rest of the model is frozen.
   */
  void set_phase(int phase);
  /** Set the greedy layerwise autoencover to a sequential model. */
  void restore_sequential_model();

  /** Forward prop step. */
  void forward_prop(execution_mode mode) override;
  /** Backward prop step. */
  void backward_prop() override;

};

}  // namespace lbann

#endif  // LBANN_MODEL_GREEDY_LAYERWISE_AUTOENCODER_HPP
