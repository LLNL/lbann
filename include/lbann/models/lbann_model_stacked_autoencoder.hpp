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
#include "lbann/layers/lbann_target_layer_unsupervised.hpp"
//#include "lbann/lbann.hpp"
#include <vector>
#include <string>

namespace lbann
{
  class stacked_autoencoder : public sequential_model
  {
  public:
    /// Constructor
    stacked_autoencoder(uint mini_batch_size,
                        lbann_comm* comm,
                        objective_fn* obj_fn,
                        layer_factory* _layer_fac,
                        Optimizer_factory* _optimizer_fac);

    /// Destructor
    ~stacked_autoencoder();

    void begin_stack(const std::string layer_name,
                     int layer_dim,
                     activation_type activation=activation_type::RELU,
                     weight_initialization init=weight_initialization::glorot_uniform,
                     std::vector<regularizer*> regularizers={});

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
    bool train_mini_batch(long *num_samples, long *num_errors);

    /// Evaluate neural network
    DataType evaluate(execution_mode mode=execution_mode::testing) { }
    /// Evaluation step on one mini-batch
    bool evaluate_mini_batch(long *num_samples, long *num_errors) { }

    /// Reconstruction uses unsupervised target layer
    DataType reconstruction();

    bool reconstruction_mini_batch(long *num_samples, long *num_errors);

    /// Get train accuracy
    /** Classification accuracy over the last training epoch
     */
    DataType get_train_accuracy() const { return m_train_accuracy; }
    /// Get validation accuracy
    DataType get_validate_accuracy() const { return m_validation_accuracy; }
    /// Get test accuracy
    DataType get_test_accuracy() const { return m_test_accuracy; }

    /// Get test accuracy
    DataType get_reconstruction_accuracy() const { return m_reconstruction_accuracy; }

    /// Returns the model's name
    const std::string & name() { return m_name; }

    //vector<Layer>& get_layers() const {return m_layers;}

  protected:
    /// Train accuracy over last training epoch
    DataType m_train_accuracy;
    /// Validation accuracy
    DataType m_validation_accuracy;
    /// Test accuracy
    DataType m_test_accuracy;
    /// Reconstruction accuracy
    DataType m_reconstruction_accuracy;
    size_t m_num_layers;
    target_layer_unsupervised* m_target_layer;
    /// the Model's name
    std::string m_name;
  };
}


#endif // LBANN_MODEL_STACKED_AUTOENCODER_HPP
