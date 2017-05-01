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
#include "lbann/layers/lbann_layer_reconstruction.hpp"
#include "lbann/data_readers/lbann_image_utils.hpp"

using namespace std;
using namespace El;



lbann::greedy_layerwise_autoencoder::greedy_layerwise_autoencoder(const uint mini_batch_size,
                                                lbann_comm* comm,
                                                objective_functions::objective_fn* obj_fn,
                                                layer_factory* _layer_fac,
                                                optimizer_factory* _optimizer_fac)
  : sequential_model(mini_batch_size, comm, obj_fn, _layer_fac, _optimizer_fac),
    m_have_mirror(0),m_phase_end(2) {}

lbann::greedy_layerwise_autoencoder::~greedy_layerwise_autoencoder() {}

struct lbann_model_greedy_layerwise_autoencoder_header {
    uint32_t phase_index; //should be m_current_phase??
    uint32_t have_mirror;
};

void lbann::greedy_layerwise_autoencoder::reset_phase() {
  m_current_phase = 0;
  m_current_epoch = 0;
  m_layers.resize(m_layers.size()-m_reconstruction_layers.size());
  //clear m_reconstruction layers
  m_reconstruction_layers.clear();
}
  
bool lbann::greedy_layerwise_autoencoder::save_to_checkpoint_shared(lbann::persist& p)
{
    // have rank 0 write record whether we have a mirror layer inserted
    // we do this first, because we need to insert it again when reading back
    if (p.m_rank == 0) {
        p.write_uint32(persist_type::train, "gla_phase_index", (uint32_t) m_current_phase);
        p.write_uint32(persist_type::train, "gla_have_mirror", (uint32_t) m_have_mirror);
    }

    // write parameters from base class first
    sequential_model::save_to_checkpoint_shared(p);

    return true;
}

bool lbann::greedy_layerwise_autoencoder::load_from_checkpoint_shared(lbann::persist& p)
{
    // have rank 0 read whether we have a mirror layer inserted
    struct lbann_model_greedy_layerwise_autoencoder_header header;
    if (p.m_rank == 0) {
        p.read_uint32(persist_type::train, "gla_phase_index", &header.phase_index);
        p.read_uint32(persist_type::train, "gla_have_mirror", &header.have_mirror);
    }

    // TODO: this assumes homogeneous processors
    // broadcast state from rank 0
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    // insert the mirror layer if needed
    uint32_t phase_index = header.phase_index;
    uint32_t have_mirror = header.have_mirror;
    if (have_mirror) {
      // note that this calls setup on the layers,
      // and setup reinitializes a bunch of values like data reader positions
      // and optimization layer cache values that we'll overwrite
      // in load_from_checkpoint_shared below
      insert_mirror(phase_index);
    }

    // read parameters from base class first
    sequential_model::load_from_checkpoint_shared(p);

    return true;
}

void lbann::greedy_layerwise_autoencoder::summarize(lbann_summary& summarizer) {
  for (size_t l = 1; l < m_layers.size(); ++l) {
    m_layers[l]->summarize(summarizer, get_cur_step());
  }
}

// inserts a mirror layer for specified layer index
void lbann::greedy_layerwise_autoencoder::insert_mirror(uint32_t layer_index)
{
  // compute layer index for mirrror
  size_t mirror_index = layer_index + 2;

  // build mirror layer
  Layer* original_layer = m_layers[layer_index];
  optimizer *opt = create_optimizer();
  reconstruction_layer* mirror_layer = new reconstruction_layer(original_layer->m_data_layout, mirror_index, comm, opt, m_mini_batch_size, original_layer);

  // insert mirror layer into model
  insert(mirror_index, mirror_layer);

  //call base model set up at each phase to reindex and set appropriate matrices, fp and bp input
  //assume that necessary layer parameters are set e.g., NumNeurons when layers were constructed
  setup(layer_index, mirror_index+1);  //set up  all active layers

  // set flag to indicate that we have a mirror layer inserted
  m_have_mirror = 1;
}

// removes a mirror layer for specified layer index
void lbann::greedy_layerwise_autoencoder::remove_mirror(uint32_t layer_index)
{
  if (m_have_mirror) {
    // compute layer index for mirrror
    size_t mirror_index = layer_index + 2;

    // drop the mirror layer from the model
    remove(mirror_index); ///any delete on heap, vector resize?

    // call base model setup again to reindex and set appropriate fp and bp input
    if (comm->am_world_master()) {
      std::cout << "Phase [" << layer_index << "] Done, Reset Layers " << std::endl;
      for(auto& l:m_layers) std::cout << "Layer [ " << l->Index << "] #NumNeurons: " << l->NumNeurons << std::endl;
    }
    setup();

    // set flag to indicate we've deleted our mirror layer
    m_have_mirror = 0;
  }
}

void lbann::greedy_layerwise_autoencoder::train(int num_epochs, int evaluation_frequency)
{
  size_t num_phases = m_layers.size()-1;
  // get to training, layer by layer
  while(m_current_phase < num_phases){
    //m_current_phase = phase_index;
    m_phase_end = m_current_phase+2;
    Layer* original_layer = m_layers[m_current_phase];
    optimizer *opt = create_optimizer();
    reconstruction_layer*  mirror_layer = new reconstruction_layer(original_layer->m_data_layout, m_phase_end, comm, opt, m_mini_batch_size,original_layer);
    Layer* tmp;
    //if not at the last layer/phase, swap otherwise insert new
    if(m_current_phase < num_phases-1) tmp = swap(m_phase_end,mirror_layer);
    else  insert(m_phase_end,mirror_layer);
    //call base model set up at each phase to reindex and set appropriate matrices, fp and bp input
    //assume that necessary layer parameters are set e.g., NumNeurons when layers were constructed
    setup(m_phase_end,m_phase_end+1);  //set up just the added (new) layers
    train_phase(num_epochs,evaluation_frequency);

    if (comm->am_world_master()) {
      //end of phase cbs e.g., save a number of image to file
      do_phase_end_cbs();
    }
    m_reconstruction_layers.insert(m_reconstruction_layers.begin(),mirror_layer);
    //swap back
    if(m_current_phase < num_phases-1) swap(m_phase_end,tmp);

    // move on to the next phase
    m_current_phase++;
  }
  //evaluate and save all layers for i.e., (1) global cost (2) image reconstruction
  evaluate(execution_mode::testing);

}

/*void lbann::greedy_layerwise_autoencoder::train(int num_epochs, int evaluation_frequency)
{
  // compute number of layers we need to train
  size_t num_phases = m_layers.size() - 1;
  if (m_have_mirror) {
    // already have a mirror layer loaded, subtract that off
    num_phases--;
  }

  // get to training, layer by layer
  while(m_current_phase < num_phases){
    // add mirror layer for training
    // (may already have this after loading checkpoint)
    if (! m_have_mirror) {
      insert_mirror(m_current_phase);
    }

    //debug
    train_phase(m_current_phase, num_epochs, evaluation_frequency);

    if (comm->am_world_master()) {
      //end of phase cbs e.g., save a number of image to file
      do_phase_end_cbs();
    }

    // drop mirror layer
    remove_mirror(m_current_phase);

    // move on to the next phase
    m_current_phase++;
  }
}*/

void lbann::greedy_layerwise_autoencoder::train_phase(int num_epochs, int evaluation_frequency)
{
  do_train_begin_cbs();

  // Epoch main loop
  while (get_cur_epoch() < num_epochs) {
    // Check if training has been terminated
    if (get_terminate_training()) break;

    // due to restart, may not always be at start of epoch
    // use mini batch index in data reader to signify start of epoch
    if (at_epoch_start()) {
      ++m_current_epoch;
      do_epoch_begin_cbs(); // needed for selected callback e.g., dump matrices
    }

    //Overide default print callback
    if (comm->am_world_master()) {
      //std::cout << "-----------------------------------------------------------" << std::endl;
      //std::cout << "Phase [" << m_current_phase  << "] Epoch [" << m_current_epoch << "]" <<  std::endl;
      std::cout << "\n Training hidden layer [" << m_current_phase+1  << "] at layer-wise epoch [" << m_current_epoch << "]" <<  std::endl;
      std::cout << "-----------------------------------------------------------" << std::endl;
    }

    /// Set the execution mode to training
    m_execution_mode = execution_mode::training;
    for (size_t l =0; l < m_layers.size(); l++) {
      m_layers[l]->m_execution_mode = execution_mode::training;
    }

    // Train on mini-batches until data set is traversed
    // Note: The data reader shuffles the data after each epoch
    for (auto&& m : metrics) { m->reset_metric(); }
    bool finished_epoch;
    do {
      finished_epoch = train_mini_batch();

      // save a checkpoint if needed
      if (need_checkpoint()) {
        checkpointShared();
      }
    } while(!finished_epoch);


    //print training reconstruction cost
    if (comm->am_world_master()) std::cout << "Layer-wise training ";
    m_layers[m_phase_end]->epoch_print();


    do_epoch_end_cbs(); //needed for selected callback e.g., dump matrices

    for (Layer* layer : m_layers) {
      layer->epoch_reset();
    } // train epoch end, this reset cost

    evaluate_phase(execution_mode::validation);

    //print validation reconstruction cost 
    if (comm->am_world_master()) std::cout << "Layer-wise validation ";
    m_layers[m_phase_end]->epoch_print();

    //Reset cost again
    for (Layer* layer : m_layers) {
      layer->epoch_reset();
    } // train epoch

    // Reset execution mode back to training
    m_execution_mode = execution_mode::training;
    for (Layer* layer : m_layers) {
      layer->m_execution_mode = execution_mode::training;
    }

    // save checkpoint after epoch
    if (need_checkpoint()) {
      checkpointShared();
    }
  }

  do_train_end_cbs();
  m_current_epoch = 0; //reset epoch counter
}

bool lbann::greedy_layerwise_autoencoder::train_mini_batch()
{
  do_batch_begin_cbs();

  // Forward propagation
  do_model_forward_prop_begin_cbs();
  //@todo; optimize this? change start index from 0 to phase_index
  for (size_t l = 0; l <= m_phase_end; ++l) {
    do_layer_forward_prop_begin_cbs(m_layers[l]);
    m_layers[l]->forwardProp();
    do_layer_forward_prop_end_cbs(m_layers[l]);
  }
  do_model_forward_prop_end_cbs();

  ++m_current_step;

  // Backward propagation
  do_model_backward_prop_begin_cbs();
  //@todo; optimize to backprop up to phase_index and not 0
  for (size_t l = m_phase_end+1; l-- > 0;) {
    do_layer_backward_prop_begin_cbs(m_layers[l]);
    m_layers[l]->backProp();
    do_layer_backward_prop_end_cbs(m_layers[l]);
  }
  do_model_backward_prop_end_cbs();

  /// Update (active) layers
  ///Freeze inactive layers
  for (size_t l = m_phase_end; l > m_current_phase; --l) {
    m_layers[l]->update();
  }
  const bool data_set_processed = m_layers[0]->update();

  do_batch_end_cbs();
  return data_set_processed;
}

void lbann::greedy_layerwise_autoencoder::evaluate_phase(execution_mode mode)
{
  // Set the execution mode
  m_execution_mode = mode;
  for (size_t l = 0; l < m_layers.size(); ++l) {
    m_layers[l]->m_execution_mode = mode;
  }

  // Evaluate on mini-batches until data set is traversed
  // Note: The data reader shuffles the data after each epoch
  for (auto&& m : metrics) { m->reset_metric(); }
  bool finished_epoch;
  do {
    finished_epoch = evaluate_mini_batch();
  } while(!finished_epoch);


  /*for (Layer* layer : m_layers) {
    layer->epoch_reset();
  }*/

  return;
}

bool lbann::greedy_layerwise_autoencoder::evaluate_mini_batch()
{
  // forward propagation (mini-batch)
  for (size_t l = 0; l < m_layers.size(); l++) {
    m_layers[l]->forwardProp();
  }

  // Update layers
  // Note: should only affect the input and target 
  // @todo: delete after check with input layer
  for (size_t l = m_phase_end; l > m_current_phase; --l) {
    m_layers[l]->update();
  }
  const bool data_set_processed = m_layers[0]->update();
  return data_set_processed;
}


void lbann::greedy_layerwise_autoencoder::evaluate(execution_mode mode)
{
  //concatenate original layers with mirror layers 
  m_layers.insert(std::end(m_layers), std::begin(m_reconstruction_layers)+1,std::end(m_reconstruction_layers));
  
  //Set appropriate layer indices and fp_input
  size_t mls = m_layers.size();
  size_t mrs_index = mls-m_reconstruction_layers.size()+1; //reconstruction layers start index
  for(size_t l = mrs_index; l < mls; ++l) m_layers[l]->Index = l;
  set_fp_input(mrs_index,mls);
  
  //@todo loop for epochs??
  m_phase_end = mls-1;
  evaluate_phase(mode);
  
  if (comm->am_world_master()) std::cout << "Global (rel. to all (in + hidden) layers) testing ";
    m_layers[m_phase_end]->epoch_print();

  for (Layer* layer : m_layers) {
    layer->epoch_reset();
  }
  
  //@todo: finetune only up to the true layers skipping the reconstruction layers
  //m_layers.resize(m_layers.size()-m_reconstruction_layers.size());
  //clear m_reconstruction layers
  //m_reconstruction_layers.clear();

  return;
}
