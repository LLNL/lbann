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

#ifndef LBANN_MODEL_STACKED_AUTOENCODER_HPP
#define LBANN_MODEL_STACKED_AUTOENCODER_HPP

#include "lbann/models/lbann_model_sequential.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include "lbann/layers/lbann_layer_reconstruction.hpp"
//#include "lbann/lbann.hpp"
#include <vector>
#include <string>

namespace lbann {
class stacked_autoencoder : public sequential_model {
 public:
  /// Constructor
  stacked_autoencoder(uint mini_batch_size,
                      lbann_comm *comm,
                      objective_functions::objective_fn *obj_fn,
                      optimizer_factory *_optimizer_fac);

  /// Destructor
  ~stacked_autoencoder();

  void begin_stack(Layer* new_layer);
  
  //void setup();
  /// Compute layer summaries
  void summarize(lbann_summary& summarizer);

  /// pre train stacked autoencoder neural network
  /** Half of the layers is pretrained and the remaining ones
   * are initialized with the transpose of the other layer W^1= W^k^T
   * @param num_epochs Number of epochs to train
   */
  void train(int num_epochs, int evaluation_frequency=0);
  /// Training step on one mini-batch
  bool train_mini_batch();

  /// Evaluate neural network
  void evaluate(execution_mode mode=execution_mode::testing) { }
  /// Evaluation step on one mini-batch
  bool evaluate_mini_batch() {
    return false;
  }

  /// Reconstruction uses unsupervised target layer
  void reconstruction();
  bool reconstruction_mini_batch();

  /// Returns the model's name
  const std::string& name() {
    return m_name;
  }

  //vector<Layer>& get_layers() const {return m_layers;}

 protected:
  size_t m_num_layers;
  reconstruction_layer<data_layout> *m_target_layer;
  /// the Model's name
  std::string m_name;
};
}


#endif // LBANN_MODEL_STACKED_AUTOENCODER_HPP
