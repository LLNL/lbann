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
                                                objective_fn* obj_fn,
                                                layer_factory* _layer_fac,
                                                Optimizer_factory* _optimizer_fac)
  : sequential_model(mini_batch_size, comm, obj_fn, _layer_fac, _optimizer_fac),
    m_train_accuracy(0.0),
    m_validation_accuracy(0.0),
    m_test_accuracy(0.0) {}

lbann::greedy_layerwise_autoencoder::~greedy_layerwise_autoencoder() {}

void lbann::greedy_layerwise_autoencoder::summarize(lbann_summary& summarizer) {
  for (size_t l = 1; l < m_layers.size(); ++l) {
    m_layers[l]->summarize(summarizer, get_cur_step());
  }
}


void lbann::greedy_layerwise_autoencoder::train(int num_epochs, int evaluation_frequency)
{
  size_t num_phases = m_layers.size()-1;
  for(size_t phase_index=0; phase_index < num_phases; ++phase_index){
    size_t phase_end = phase_index+2;
    Layer* original_layer = m_layers[phase_index];
    Optimizer *optimizer = optimizer_fac->create_optimizer();
    target_layer_unsupervised*  mirror_layer = new target_layer_unsupervised(phase_end, comm, optimizer, m_mini_batch_size,original_layer);
    insert(phase_end,mirror_layer);
    //call base model set up at each phase to reindex and set appropriate matrices, fp and bp input
    //assume that necessary layer parameters are set e.g., NumNeurons when layers were constructed
    setup(phase_index,phase_end+1);  //set up  all active layers
    //debug
    train_phase(phase_index, num_epochs,evaluation_frequency);
    remove(phase_end); ///any delete on heap, vector resize?
    //call base model setup again to reindex and set appropriate fp and bp input
    if (comm->am_world_master()) {
      std::cout << "Phase [" << phase_index << "] Done, Reset Layers " << std::endl;
      for(auto& l:m_layers) std::cout << "Layer [ " << l->Index << "] #NumNeurons: " << l->NumNeurons << std::endl;
    }
    setup();

  }

}

void lbann::greedy_layerwise_autoencoder::train_phase(size_t phase_index, int num_epochs, int evaluation_frequency)
{
  size_t phase_end = phase_index+2;
  do_train_begin_cbs();

  // Epoch main loop
  for (int epoch = 0; epoch < num_epochs; ++epoch) {

    // Check if training has been terminated
    if (get_terminate_training()) break;

    ++m_current_epoch;
    do_epoch_begin_cbs(); // needed for selected callback e.g., dump matrices
    //Overide default print callback
    if (comm->am_world_master()) {
      std::cout << "-----------------------------------------------------------" << std::endl;
      std::cout << "Phase [" << phase_index  << "] Epoch [" << epoch << "]" <<  std::endl;
      std::cout << "-----------------------------------------------------------" << std::endl;
    }

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


    //Is training and testing enough? Do we need validation? val/test look similar!
    /*if(evaluation_frequency > 0
       && (epoch + 1) % evaluation_frequency == 0) {
      m_validation_accuracy = evaluate(execution_mode::validation);

      // Set execution mode back to training
      m_execution_mode = execution_mode::training;
      for (Layer* layer : m_layers) {
        layer->m_execution_mode = execution_mode::training;
      }
    }*/

    //print training reconstruction cost
    if (comm->am_world_master()) std::cout << "Training ";
    m_layers[phase_end]->epoch_print();
    do_epoch_end_cbs(); //needed for selected callback e.g., dump matrices
    for (Layer* layer : m_layers) {
      layer->epoch_reset();
    } // train epoch end, this reset cost

    evaluate(execution_mode::testing);

    //print testing reconstruction cost (somewhat validation)
    if (comm->am_world_master()) std::cout << "Testing ";
    m_layers[phase_end]->epoch_print();
    //Reset cost again
    for (Layer* layer : m_layers) {
      layer->epoch_reset();
    } // train epoch
    // Reset execution mode back to training
    m_execution_mode = execution_mode::training;
    for (Layer* layer : m_layers) {
      layer->m_execution_mode = execution_mode::training;
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
  // Set the execution mode
  m_execution_mode = mode;
  for (size_t l = 0; l < m_layers.size(); ++l) {
    m_layers[l]->m_execution_mode = mode;
  }

  // Evaluate on mini-batches until data set is traversed
  // Note: The data reader shuffles the data after each epoch
  long num_samples = 0; //not use
  long num_errors = 0; //not use
  bool finished_epoch;
  do {
    finished_epoch = evaluate_mini_batch(&num_samples, &num_errors);
  } while(!finished_epoch);


  /*for (Layer* layer : m_layers) {
    layer->epoch_reset();
  }*/

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
  //*num_errors += (long) L2NormSum;
  //*num_samples += m_mini_batch_size;

  // Update layers
  // Note: should only affect the input and target layers
  for (size_t l = m_layers.size() - 1; l > 0; --l) {
    m_layers[l]->update();
  }
  const bool data_set_processed = m_layers[0]->update();
  return data_set_processed;
}
