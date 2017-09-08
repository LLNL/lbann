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
// lbann_model_sequential .hpp .cpp - Sequential neural network models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_SEQUENTIAL_HPP
#define LBANN_MODEL_SEQUENTIAL_HPP

#include "lbann/models/model.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/layers/activations/activation.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/io/persist.hpp"
#include <vector>
#include <string>

namespace lbann {

class sequential_model : public model {
 public:

  /// Constructor
  sequential_model(int mini_batch_size,
                   lbann_comm *comm,
                   objective_functions::objective_function *obj_fn,
                   optimizer_factory *optimizer_fac);
  sequential_model(const sequential_model& other);
  sequential_model& operator=(const sequential_model& other);

  /// Destructor
  virtual ~sequential_model();

  /// Save model to file
  /** @todo This is old and likely broken */
  bool save_to_file(const std::string file_dir);
  /// Load model from file
  /** @todo This is old and likely broken */
  bool load_from_file(const std::string file_dir);

  /// Save model to checkpoint
  /** @todo This is old and likely broken */
  bool save_to_checkpoint(int fd, const char *filename, size_t *bytes);
  /// Load model from checkpoint
  /** @todo This is old and likely broken */
  bool load_from_checkpoint(int fd, const char *filename, size_t *bytes);

  bool save_to_checkpoint_shared(persist& p);
  bool load_from_checkpoint_shared(persist& p);

  /// Get list of layers
  virtual std::vector<Layer *>& get_layers() {
    return m_layers;
  }

  /// Set layers
  virtual void set_layers(vector<Layer *>& layers) {
    m_layers = layers;
  }

 
  /// Add layer to sequential model
  /** @todo Consider removing this function. The destructor
   *  deallocates all layers, so we might run into problems if a
   *  layer is deallocated externally. */
  virtual int add(Layer *new_layer);

  /// Remove layer from sequential model
  /** @todo This will mess up layer indices */
  virtual void remove(int index);

  /// Insert layer in sequential model
  /** @todo This will mess up layer indices.
   *  @todo Consider removing this function. The destructor
   *  deallocates all layers, so we might run into problems if a
   *  layer is deallocated externally. */
  virtual void insert(int index, Layer *new_layer);

  /// Replace layer in sequential model
  virtual Layer *swap(int index, Layer *new_layer);

  /// Setup sequential model
  virtual void setup(int start_index=0, int end_index=0);

  /// Train model
  virtual void train(int num_epochs) = 0;
  /// Training step on one mini-batch
  virtual bool train_mini_batch() = 0;

  /** Return true if about to start a new training epoch */
  virtual bool at_epoch_start();

  /** Check if the model has a valid data set for the execution mode */
  virtual bool is_execution_mode_valid(execution_mode mode);

  /// Evaluate model
  virtual void evaluate(execution_mode mode) = 0;
  /// Evaluation step on one mini-batch
  virtual bool evaluate_mini_batch() = 0;

  /// returns the number of neurons in the most recently added layer, or -1
  /// if there is none
  int num_previous_neurons();

 protected:
  /// List of layers
  std::vector<Layer *> m_layers;

};

}  // namespace lbann

#endif  // LBANN_MODEL_SEQUENTIAL_HPP
