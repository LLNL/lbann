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

#include "lbann/models/lbann_model_greedy_layerwise_autoencoder.hpp"
#include "lbann/layers/lbann_target_layer_unsupervised.hpp"

using namespace std;
using namespace El;



lbann::greedy_layerwise_autoencoder::greedy_layerwise_autoencoder(const uint mini_batch_size,
                                                lbann_comm* comm,
                                                layer_factory* _layer_fac,
                                                Optimizer_factory* _optimizer_fac)
  : sequential_model(mini_batch_size, comm, _layer_fac, _optimizer_fac),
    m_train_accuracy(0.0),
    m_validation_accuracy(0.0),
    m_test_accuracy(0.0) {}

lbann::greedy_layerwise_autoencoder::~greedy_layerwise_autoencoder() {}

//delete??
void lbann::greedy_layerwise_autoencoder::check_gradient(CircMat& X, CircMat& Y, double* gradient_errors)
{
  // setup input (last/additional row should always be 1)
  Copy(X, *(m_layers[0]->Acts));

  // forward propagation (mini-batch)
  DataType L2NormSum = 0;
  for (size_t l = 1; l < m_layers.size(); l++)
    L2NormSum = m_layers[l]->forwardProp(L2NormSum);

  // backward propagation (mini-batch)
  for (size_t l = m_layers.size() - 1; l >= 1; l--) {
    m_layers[l]->backProp();
  }

  // check gradient
  gradient_errors[0] = 0;
  for (size_t l = 1; l < m_layers.size(); l++)
    gradient_errors[l] = m_layers[l]->checkGradientMB(*m_layers[l-1]);
}

void lbann::greedy_layerwise_autoencoder::summarize(lbann_summary& summarizer) {
  for (size_t l = 1; l < m_layers.size(); ++l) {
    m_layers[l]->summarize(summarizer, get_cur_step());
  }
}


void lbann::greedy_layerwise_autoencoder::train(int num_epochs, int evaluation_frequency)
{
  size_t num_phases = m_layers.size();
  for(size_t phase_index=0; phase_index < num_phases; ++phase_index){
    Layer* sibling_layer = m_layers[phase_index];
    //int num_prev_neurons = m_layers[phase_index+1]->NumNeurons;
    Optimizer *optimizer = optimizer_fac->create_optimizer();
    target_layer_unsupervised*  mirror_layer = new target_layer_unsupervised(phase_index+2, comm, optimizer, m_mini_batch_size,sibling_layer);
    //mirror_layer.setup(num_prev_neurons); //bug ? adding layer after network setup?
    insert(phase_index+2,mirror_layer);
    //rewire_index(); // move to base or inside insert
    //call base model set up again to reindex and set appropriate fp and bp input
    //@todo: this can be optimized by giving appropriate start index
    //assume that necessary layer parameters are set e.g., NumNeurons
    setup();  //set up all layers/ or all active layers
    //debug
    for(auto& l:m_layers) std::cout << "Layer [ " << l->Index << "] #NumNeurons: " << l->NumNeurons << std::endl;
    train_phase(phase_index, num_epochs,evaluation_frequency);
    remove(phase_index+2); ///any delete on heap, vector resize?
    //call base model setup again to reindex and set appropriate fp and bp input
    //@todo: this can be optimized by giving appropriate start index
    if(phase_index < num_phases-1) setup(); //skip last round, rewrite

  }

}

void lbann::greedy_layerwise_autoencoder::train_phase(size_t phase_index, int num_epochs, int evaluation_frequency)
{
  //size_t phase_end = phase_index+2;
  do_train_begin_cbs();

  // Epoch main loop
  for (int epoch = 0; epoch < num_epochs; ++epoch) {

    // Check if training has been terminated
    if (get_terminate_training()) break;

    ++m_current_epoch;
    do_epoch_begin_cbs();

    /// Set the execution mode to training
    m_execution_mode = execution_mode::training;
    for (size_t l =0; l < m_layers.size(); l++) {
      m_layers[l]->m_execution_mode = execution_mode::training;
    }

    // Train on mini-batches until data set is traversed
    // Note: The data reader shuffles the data after each epoch
    long num_samples = 0;
    long num_errors = 0;
    bool finished_epoch;
    do {
      finished_epoch = train_mini_batch(phase_index, &num_samples, &num_errors);
    } while(!finished_epoch);

    // Compute train accuracy on current epoch
    m_train_accuracy = DataType(num_samples - num_errors) / num_samples * 100;

    /*if(evaluation_frequency > 0
       && (epoch + 1) % evaluation_frequency == 0) {
      // Evaluate model on validation set
      // TODO: do we need validation callbacks here?
      // do_validation_begin_cbs();
      m_validation_accuracy = evaluate(execution_mode::validation);
      // do_validation_end_cbs();

      // Set execution mode back to training
      m_execution_mode = execution_mode::training;
      for (size_t l = ; l < m_layers.size(); l++) {
        m_layers[l]->m_execution_mode = execution_mode::training;
      }
    }*/

    do_epoch_end_cbs();
    for (Layer* layer : m_layers) {
      layer->epoch_reset();
    }
  }
  do_train_end_cbs();
}

bool lbann::greedy_layerwise_autoencoder::train_mini_batch(size_t phase_index, long *num_samples,
                                                  long *num_errors)
{
  size_t phase_end = phase_index+2;
  do_batch_begin_cbs();

  // Forward propagation
  do_model_forward_prop_begin_cbs();
  DataType L2NormSum = 0;
  //@todo; optimize this? change start index from 0 to phase_index
  for (size_t l = 0; l <= phase_end; ++l) {
    do_layer_forward_prop_begin_cbs(m_layers[l]);
    L2NormSum = m_layers[l]->forwardProp(L2NormSum);
    do_layer_forward_prop_end_cbs(m_layers[l]);
  }
  *num_errors += (long) L2NormSum;
  *num_samples += m_mini_batch_size;
  do_model_forward_prop_end_cbs();

  // Update training accuracy
  m_train_accuracy = DataType(*num_samples - *num_errors) / *num_samples * 100;
  ++m_current_step;

  // Backward propagation
  do_model_backward_prop_begin_cbs();
  //@todo; optimize to backprop up to phase_index and not 0
  for (size_t l = phase_end+1; l-- > 0;) {
    do_layer_backward_prop_begin_cbs(m_layers[l]);
    m_layers[l]->backProp();
    do_layer_backward_prop_end_cbs(m_layers[l]);
  }
  do_model_backward_prop_end_cbs();

  /// Update (active) layers
  ///Freeze inactive layers
  for (size_t l = phase_end; l > phase_index; --l) {
    m_layers[l]->update();
  }
  const bool data_set_processed = m_layers[0]->update();

  do_batch_end_cbs();
  return data_set_processed;
}

DataType lbann::greedy_layerwise_autoencoder::evaluate(execution_mode mode)
{
  switch(mode) {
  case execution_mode::validation:
    do_validation_begin_cbs(); break;
  case execution_mode::testing:
    do_test_begin_cbs(); break;
  default:
    throw lbann_exception("Illegal execution mode in evaluate function");
  }

  // Set the execution mode
  m_execution_mode = mode;
  for (size_t l = 0; l < m_layers.size(); ++l) {
    m_layers[l]->m_execution_mode = mode;
  }

  // Evaluate on mini-batches until data set is traversed
  // Note: The data reader shuffles the data after each epoch
  long num_samples = 0;
  long num_errors = 0;
  bool finished_epoch;
  do {
    finished_epoch = evaluate_mini_batch(&num_samples, &num_errors);
  } while(!finished_epoch);

  // Compute test accuracy
  m_test_accuracy = DataType(num_samples - num_errors) / num_samples * 100;

  switch(mode) {
  case execution_mode::validation:
    do_validation_end_cbs(); break;
  case execution_mode::testing:
    do_test_end_cbs();
    // Reset after testing.
    for (Layer* layer : m_layers) {
      layer->epoch_reset();
    }
    break;
  default:
    throw lbann_exception("Illegal execution mode in evaluate function");
  }

  return m_test_accuracy;
}

bool lbann::greedy_layerwise_autoencoder::evaluate_mini_batch(long *num_samples,
                                                     long *num_errors)
{
  // forward propagation (mini-batch)
  DataType L2NormSum = 0;
  for (size_t l = 0; l < m_layers.size(); l++) {
    L2NormSum = m_layers[l]->forwardProp(L2NormSum);
  }
  *num_errors += (long) L2NormSum;
  *num_samples += m_mini_batch_size;

  // Update layers
  // Note: should only affect the input and target layers
  for (size_t l = m_layers.size() - 1; l > 0; --l) {
    m_layers[l]->update();
  }
  const bool data_set_processed = m_layers[0]->update();
  return data_set_processed;
}

//remove, not use
void lbann::greedy_layerwise_autoencoder::rewire_index()
{
  for (size_t n = 0; n < m_layers.size(); n++) {
    Layer* layer = m_layers[n];
    layer->Index = n;
  }
}
