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

#ifndef LBANN_MODEL_DNN_HPP
#define LBANN_MODEL_DNN_HPP

#include "lbann/models/lbann_model_sequential.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include <vector>
#include <string>

namespace lbann
{
  class deep_neural_network : public sequential_model
  {
  public:
    /// Constructor
    deep_neural_network(uint mini_batch_size,
                        lbann_comm* comm, 
                        objective_fn* obj_fn,
                        layer_factory* _layer_fac,
                        Optimizer_factory* _optimizer_fac);
    
    /// Destructor
    ~deep_neural_network();

    /// Check error in gradients
    /** @todo This is very old and probably broken
     */
    void check_gradient(CircMat& X, CircMat& Y, double* gradient_errors);

    /// Compute layer summaries
    void summarize(lbann_summary& summarizer);

    /// Train neural network
    /** @param num_epochs Number of epochs to train
     *  @param evaluation_frequency How often to evaluate model on
     *  validation set. A value less than 1 will disable evaluation.
     */
    void train(int num_epochs, int evaluation_frequency=0);
    /// Training step on one mini-batch
    bool train_mini_batch(long *num_samples, long *num_errors);

    /// Evaluate neural network
    DataType evaluate(execution_mode mode=execution_mode::testing);
    /// Evaluation step on one mini-batch
    bool evaluate_mini_batch(long *num_samples, long *num_errors);

    /// Get train accuracy
    /** Classification accuracy over the last training epoch
     */
    DataType get_train_accuracy() const { return m_train_accuracy; }
    /// Get validation accuracy
    DataType get_validate_accuracy() const { return m_validation_accuracy; }
    /// Get test accuracy
    DataType get_test_accuracy() const { return m_test_accuracy; }

    /// Returns the model's name
    const string & name() { return m_name; }

  protected:
    /// Train accuracy over last training epoch
    DataType m_train_accuracy;
    /// Validation accuracy
    DataType m_validation_accuracy;
    /// Test accuracy
    DataType m_test_accuracy;
    ///string name
    std::string m_name;
  };
}


#endif // LBANN_MODEL_DNN_HPP
