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
#include "lbann/layers/activations/activation.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/io/persist.hpp"
#include <vector>
#include <string>

namespace lbann {

class planar_model : public model {
 public:

  /// Constructor
  planar_model(int mini_batch_size,
                   lbann_comm *comm,
                   objective_functions::objective_fn *obj_fn,
                   optimizer_factory *optimizer_fac);
/**
  planar_model(const planar_model& other);
  planar_model& operator=(const planar_model& other); */

  /// Destructor
  ~planar_model();

  /**
   * Following functions are removed from the original model_sequential.hpp
   * They are old anyways, and I don't see any reason to include these defunct 
   * functions. */
  /// Save model to file
  /** @todo This is old and likely broken */
  //bool save_to_file(const std::string file_dir);
  /// Load model from file
  /** @todo This is old and likely broken */
  //bool load_from_file(const std::string file_dir);

  /// Save model to checkpoint
  /** @todo This is old and likely broken */
  //bool save_to_checkpoint(int fd, const char *filename, size_t *bytes);
  /// Load model from checkpoint
  /** @todo This is old and likely broken */
  //bool load_from_checkpoint(int fd, const char *filename, size_t *bytes);

  bool save_to_checkpoint_shared(persist& p);
  bool load_from_checkpoint_shared(persist& p);

  /// Get list of layers
  virtual std::vector<std::vector<Layer*> >& get_layers() {
    return m_layers;
  }

  /// Set layers
  virtual void set_layers(std::vector<std::vector<Layer*> >& layers) {
    m_layers = layers;
  }

 
  /** Following functions are used to add a set of layers at given horizontal level
   *  on a planar space. The layers are added either by duplicating a single layer
   *  or placing individual layers. */
  //virtual int stackup_duplicate(int horizontal_index, int num_heads, Layer *new_layer);
  //virtual int stackup(int horizontal_index, int vertical_index, Layer *new_layer);
  virtual int stackup_tail(int hindex, Layer *new_layer);
  virtual int stackup_duplicate(Layer *new_layer);

  /**
   * Following functions are removed from the original model_sequential 
   * because they are not critical. Rather focus on core functions. */
  /** @todo This will mess up layer indices */
  //virtual void remove(int index);

  /// Insert layer in planar model
  /** @todo This will mess up layer indices.
   *  @todo Consider removing this function. The destructor
   *  deallocates all layers, so we might run into problems if a
   *  layer is deallocated externally. */
  //virtual void insert(int index, Layer *new_layer);

  /// Replace layer in planar model
  //virtual Layer *swap(int index, Layer *new_layer);

  /// Setup planar model
  virtual void setup(int start_index=0, int end_index=0);

  /// Train model
  /** @param num_epochs Number of epochs to train
   *  @param evaluation_frequency How often to evaluate model on
   *  validation set. A value less than 1 will disable evaluation.
   */
  virtual void train(int num_epochs);
  /// Training step on one mini-batch
  virtual bool train_mini_batch() = 0;

  /** Return true if about to start a new training epoch */
  virtual bool at_epoch_start();

  /// Evaluate model
  virtual void evaluate(execution_mode mode) = 0;
  /// Evaluation step on one mini-batch
  virtual bool evaluate_mini_batch() = 0;

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
