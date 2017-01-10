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

#include "lbann/models/lbann_model.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include "lbann/layers/lbann_layer_activations.hpp"
#include "lbann/data_readers/lbann_data_reader.hpp"
#include "lbann/layers/lbann_layer_factory.hpp"
#include "lbann/io/lbann_persist.hpp"
#include <vector>
#include <string>

namespace lbann
{
class sequential_model : public model
  {
  public:

    /// Constructor
    sequential_model(const uint mini_batch_size,
                     lbann_comm* comm,
                     objective_fn* obj_fn,
                     layer_factory* layer_fac,
                     Optimizer_factory* optimizer_factory);

    /// Destructor
    ~sequential_model();

    /// Save model to file
    /** @todo This is old and likely broken */
    bool save_to_file(const std::string file_dir);
    /// Load model from file
    /** @todo This is old and likely broken */
    bool load_from_file(const std::string file_dir);

    /// Save model to checkpoint
    /** @todo This is old and likely broken */
    bool save_to_checkpoint(int fd, const char* filename, uint64_t* bytes);
    /// Load model from checkpoint
    /** @todo This is old and likely broken */
    bool load_from_checkpoint(int fd, const char* filename, uint64_t* bytes);

    /** @todo This is old and likely broken */
    bool save_to_checkpoint_shared(persist& p);
    /** @todo This is old and likely broken */
    bool load_from_checkpoint_shared(persist& p);

    /// Get mini-batch size
    int get_mini_batch_size() const { return m_mini_batch_size; }
    /// Get list of layers
    virtual std::vector<Layer*>& get_layers() { return m_layers; }

    /// Set layers
    virtual void set_layers(vector<Layer*>& layers) {m_layers = layers;}

    /// Add layer to sequential model
    virtual uint add(const std::string layer_name,
                     int layer_dim,
                     activation_type activation=activation_type::RELU,
                     weight_initialization init=weight_initialization::glorot_uniform,
                     std::vector<regularizer*> regularizers={});

    /// Add layer to sequential model
    /** @todo Consider removing this function. The destructor
     *  deallocates all layers, so we might run into problems if a
     *  layer is deallocated externally. */
    virtual uint add(Layer *new_layer);

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
    virtual Layer* swap(int index, Layer *new_layer);

    /// Setup sequential model
    virtual void setup(size_t start_index=0,size_t end_index=0);

    /// Train model
    /** @param num_epochs Number of epochs to train
     *  @param evaluation_frequency How often to evaluate model on
     *  validation set. A value less than 1 will disable evaluation.
     */
    virtual void train(int num_epochs, int evaluation_frequency=0) = 0;
    /// Training step on one mini-batch
    virtual bool train_mini_batch(long *num_samples, long *num_errors) = 0;

    /** Return true if about to start a new training epoch */
    virtual bool at_epoch_start();

    /// Evaluate model
    virtual DataType evaluate(execution_mode mode) = 0;
    /// Evaluation step on one mini-batch
    virtual bool evaluate_mini_batch(long *num_samples, long *num_errors) = 0;
#if 0
    /// Prediction step on one mini-batch
    /** @todo This is old and likely broken */
    virtual DistMat* predict_mini_batch(DistMat* X);
#endif

  protected:
    /// Mini-batch size (no ckpt, so user can override on restart)
    const int m_mini_batch_size;
    /// List of layers
    std::vector<Layer*> m_layers;
    /// Layer factory
    layer_factory* layer_fac;
    /// Optimizer factory
    Optimizer_factory* optimizer_fac;

  };
}
#endif  //  LBANN_MODEL_SEQUENTIAL_HPP
