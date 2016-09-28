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

#include "lbann/models/lbann_model_stacked_autoencoder.hpp"
#include "lbann/layers/lbann_layer_fully_connected.hpp"
#include "lbann/optimizers/lbann_optimizer.hpp"
#include "lbann/optimizers/lbann_optimizer_sgd.hpp"
#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"

#include <string>
#include <chrono>
#include <random>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include "mpi.h"

using namespace std;
using namespace El;


////////////////////////////////////////////////////////////////////////////////
// stacked_autoencoder
////////////////////////////////////////////////////////////////////////////////

lbann::stacked_autoencoder::stacked_autoencoder(const uint mini_batch_size,
                                                lbann_comm* comm,
                                                layer_factory* _layer_fac,
                                                Optimizer_factory* _optimizer_fac)
  : sequential_model(mini_batch_size, comm, _layer_fac, _optimizer_fac),
    m_train_accuracy(0.0),
    m_validation_accuracy(0.0),
    m_test_accuracy(0.0),
    m_reconstruction_accuracy(0.0){
    //m_target_layer = new target_layer_unsupervised(comm,mini_batch_size);
    }

lbann::stacked_autoencoder::~stacked_autoencoder() {
  //delete m_target_layer;
}


//This add hidden layers and their mirrors, input layer is added in base class?
void lbann::stacked_autoencoder::begin_stack(const std::string layer_name,
                               const int layer_dim,
                               const activation_type activation,
                               const weight_initialization init,
                               std::vector<regularizer*> regularizers){


  // get prev neurons
	//int mid = (int)Layers.size() / 2;
  int cur_size = m_layers.size();
  int mid = cur_size / 2;
  Layer* mid_layer = m_layers[mid];
	//int mid_num_neurons = mid_layer->NumNeurons;
  int prev_layer_dim = mid_layer->NumNeurons;

	if (cur_size == 1) {
		// create first hidden layer
    if(layer_name == "FullyConnected"){
      Optimizer *new_optimizer = optimizer_fac->create_optimizer();
      Layer* new_layer
        = layer_fac->create_layer<FullyConnectedLayer>("FullyConnected",cur_size,
                                                       prev_layer_dim,layer_dim,
                                                       m_mini_batch_size, activation, init,
                                                       comm,new_optimizer, regularizers);
      m_layers.push_back(new_layer);
      // create output/mirror layer
      Optimizer *mirror_optimizer = optimizer_fac->create_optimizer();
      Layer* mirror_layer
        = layer_fac->create_layer<FullyConnectedLayer>("FullyConnected",cur_size+1,
                                                       layer_dim,prev_layer_dim,
                                                       m_mini_batch_size,activation, init,
                                                       comm,mirror_optimizer,regularizers);
      m_layers.push_back(mirror_layer);
    }
	}
	else {
		// create hiden layer
    if(layer_name == "FullyConnected"){
      Optimizer *hidden_optimizer = optimizer_fac->create_optimizer();
      Layer* hidden_layer
        = layer_fac->create_layer<FullyConnectedLayer>("FullyConnected",cur_size,
                                                       prev_layer_dim,layer_dim,
                                                       m_mini_batch_size, activation, init,
                                                       comm,hidden_optimizer, regularizers);
      m_layers.insert(m_layers.begin()+ mid + 1,hidden_layer);
      // create mirror layer
      Optimizer *mirror_hidden_optimizer = optimizer_fac->create_optimizer();
      Layer* mirror_hidden_layer
        = layer_fac->create_layer<FullyConnectedLayer>("FullyConnected",cur_size+1,
                                                       layer_dim,prev_layer_dim,
                                                       m_mini_batch_size,activation, init,
                                                       comm,mirror_hidden_optimizer,regularizers);
      m_layers.insert(m_layers.begin() + mid + 2, mirror_hidden_layer);

    }
		// re-number layer index
		for (size_t n = 0; n < m_layers.size(); n++) {
			Layer* layer = m_layers[n];
			layer->Index = n;
		}
	}

  //sanity check
  if(comm->am_world_master()){
    //cout << "#Layers " << m_layers.size() << endl;
    //cout << "Layer Index and Layer Dim: " << endl;
    for(auto& layer:m_layers) cout << "[ " << layer->Index << "]" << layer->NumNeurons << endl;
  }

  m_num_layers = m_layers.size();
}

void lbann::stacked_autoencoder::summarize(lbann_summary& summarizer) {
  for (size_t l = 1; l < m_layers.size(); ++l) {
    m_layers[l]->summarize(summarizer, get_cur_step());
  }
}


/* Half of the layers is pretrained
and the remaining ones are initialized with the transpose of the other layer W^1= W^k^T
*/
void lbann::stacked_autoencoder::train(int num_epochs, int evaluation_frequency)
{
  m_execution_mode = execution_mode::training;
  //Supervised target to compute reconstruction cost
  /*m_target_layer->set_input_layer((input_layer_distributed_minibatch_parallel_io*)m_layers[0]);

  //m_target_layer->m_execution_mode = m_execution_mode;
  add(m_target_layer);
  m_target_layer->setup(m_layers[0]->NumNeurons);*/
  //replace with this
 //target_layer_unsupervised mirror_layer(phase_index+2, comm, optimizer, m_mini_batch_size,sibling_layer);

  do_train_begin_cbs();

  // Epoch main loop
  for (int epoch = 0; epoch < num_epochs; ++epoch) {

    // Check if training has been terminated
    if (get_terminate_training()) break;

    ++m_current_epoch;
    do_epoch_begin_cbs();

    /// Set the execution mode to training
    for (size_t l = 0; l < m_num_layers; ++l) {
    //for (size_t l = 0; l <= m_num_layers / 2; ++l) {
      m_layers[l]->m_execution_mode = m_execution_mode;
    }

    // Train on mini-batches until data set is traversed
    // Note: The data reader shuffles the data after each epoch
    long num_samples = 0;
    long num_errors = 0;
    bool finished_epoch;
    do {
      finished_epoch = train_mini_batch(&num_samples, &num_errors);
    } while(!finished_epoch);

    // Compute train accuracy on current epoch
    m_train_accuracy = DataType(num_samples - num_errors) / num_samples * 100;

    //Copy to (initialize) mirror layers
    for(size_t l=1; l<= m_num_layers/2; l++){
      //Copy(m_layers[l]->Acts, m_layers[m_num_layers-l]->fp_input)
      //output of reciprocating layer == input to (activation of )its succesor (l+1)
      m_layers[m_num_layers-l]->setup_fp_input(m_layers[l+1]->Acts);
    }

    //Reconstruction
    m_reconstruction_accuracy = reconstruction();
    cout << "Reconstruction accuracy " << m_reconstruction_accuracy << endl;

    do_epoch_end_cbs();
    for (Layer* layer : m_layers) {
      layer->epoch_reset();
    }
  }
  //todo: resize base m_layers
  cout << " In Train m_num_layers: " << m_num_layers << " m_layers size " << m_layers.size() << endl;
  do_train_end_cbs();
}

bool lbann::stacked_autoencoder::train_mini_batch(long *num_samples,
                                                  long *num_errors)
{
  do_batch_begin_cbs();

  // Forward propagation
  do_model_forward_prop_begin_cbs();
  DataType L2NormSum = 0;
  //pretrained half of layers
  for (size_t l = 0; l <= m_num_layers/2; ++l) {
  //for (size_t l = 0; l < m_layers.size(); ++l) {
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
  //pretrained half of layers
  //for (size_t l = (m_layers.size() / 2); l-- > 0;) {
  for (size_t l = round(m_num_layers / 2); l > 0; --l) {
    do_layer_backward_prop_begin_cbs(m_layers[l]);
    m_layers[l]->backProp();
    //Copy??
    do_layer_backward_prop_end_cbs(m_layers[l]);
  }
  do_model_backward_prop_end_cbs();

  // Update pretrained layers
  for (size_t l = round(m_num_layers / 2); l > 0; --l) {
    m_layers[l]->update();
  }
  //cout << "Samples : : " << *num_samples << endl;
  const bool data_set_processed = m_layers[0]->update();
  //cout << "data processed : " << data_set_processed << endl;
  do_batch_end_cbs();
  return data_set_processed;
}

DataType lbann::stacked_autoencoder::reconstruction()
{
  long num_samples = 0;
  long num_errors = 0;
  bool finished_epoch;
  do {
    finished_epoch = reconstruction_mini_batch(&num_samples, &num_errors);
  } while(!finished_epoch);

  // Compute reconstruction accuracy
  m_reconstruction_accuracy = DataType(num_samples - num_errors) / num_samples * 100;

  /*do_validation_end_cbs()
  // Reset after testing.
  for (Layer* layer : m_layers) {
    layer->epoch_reset();
  }*/


  return m_reconstruction_accuracy;
}

bool lbann::stacked_autoencoder::reconstruction_mini_batch(long *num_samples,
                                                     long *num_errors)
{
  // forward propagation (mini-batch)
  cout << " In Recon m_num_layers: " << m_num_layers << " m_layers size " << m_layers.size() << endl;
  DataType L2NormSum = 0;
  for (size_t l = 0; l < m_layers.size(); l++) {
    L2NormSum = m_layers[l]->forwardProp(L2NormSum);
  }
  *num_errors += (long) L2NormSum;
  *num_samples += m_mini_batch_size;

  // Update layers
  // Note: should only affect the input and target layers
  for (size_t l = m_layers.size() - 2; l > 0; --l) {
    m_layers[l]->update();
  }
  const bool data_set_processed = m_layers[0]->update();
  return data_set_processed;
}
