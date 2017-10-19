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

#include "lbann/models/model_greedy_layerwise_autoencoder.hpp"
#include "lbann/layers/io/target/reconstruction.hpp"
#include "lbann/data_readers/image_utils.hpp"

namespace lbann {

greedy_layerwise_autoencoder::greedy_layerwise_autoencoder(int mini_batch_size,
                                                           lbann_comm *comm,
                                                           objective_functions::objective_function *obj_fn,
                                                           optimizer_factory *_optimizer_fac)
  : sequential_model(mini_batch_size, comm, obj_fn, _optimizer_fac),
    m_phase_end(2), m_start_index(0), m_end_index(0), m_have_mirror(0) {}

greedy_layerwise_autoencoder::~greedy_layerwise_autoencoder() {}

struct lbann_model_greedy_layerwise_autoencoder_header {
  uint32_t phase_index; //should be m_current_phase??
  uint32_t have_mirror;
};

void greedy_layerwise_autoencoder::reset_phase() {
  m_current_phase = 0;
  m_current_epoch = 0;
  m_start_index = 0;
  m_end_index = 0;
  m_layers.resize(m_layers.size()-m_reconstruction_layers.size());
  //clear m_reconstruction layers
  m_reconstruction_layers.clear();
}

bool greedy_layerwise_autoencoder::save_to_checkpoint_shared(persist& p) {
  // have rank 0 write record whether we have a mirror layer inserted
  // we do this first, because we need to insert it again when reading back
  if (p.get_rank() == 0) {
    p.write_uint32(persist_type::train, "gla_phase_index", (uint32_t) m_current_phase);
    p.write_uint32(persist_type::train, "gla_have_mirror", (uint32_t) m_have_mirror);
  }

  // write parameters from base class first
  sequential_model::save_to_checkpoint_shared(p);

  return true;
}

bool greedy_layerwise_autoencoder::load_from_checkpoint_shared(persist& p) {
  // have rank 0 read whether we have a mirror layer inserted
  struct lbann_model_greedy_layerwise_autoencoder_header header;
  if (p.get_rank() == 0) {
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

void greedy_layerwise_autoencoder::summarize_stats(lbann_summary& summarizer) {
  for (size_t l = 1; l < m_layers.size(); ++l) {
    m_layers[l]->summarize_stats(summarizer, get_cur_step());
  }
}

void greedy_layerwise_autoencoder::summarize_matrices(lbann_summary& summarizer) {
  for (size_t l = 1; l < m_layers.size(); ++l) {
    m_layers[l]->summarize_matrices(summarizer, get_cur_step());
  }
}

// inserts a mirror layer for specified layer index
void greedy_layerwise_autoencoder::insert_mirror(uint32_t layer_index) {
  // compute layer index for mirrror
  size_t mirror_index = layer_index + 2;

  // build mirror layer
  Layer *original_layer = m_layers[layer_index];
  Layer *mirror_layer = NULL;
  switch(original_layer->get_data_layout()){
  case data_layout::MODEL_PARALLEL:
    mirror_layer = new reconstruction_layer<data_layout::MODEL_PARALLEL>(mirror_index, m_comm, original_layer);
    break;
  case data_layout::DATA_PARALLEL:
    mirror_layer = new reconstruction_layer<data_layout::DATA_PARALLEL>(mirror_index, m_comm, original_layer);
    break;
  default:
    break;
  }

  // insert mirror layer into model
  insert(mirror_index, mirror_layer);

  //call base model set up at each phase to reindex and set appropriate matrices, fp and bp input
  //assume that necessary layer parameters are set e.g., m_num_neurons when layers were constructed
  setup_subset(layer_index, mirror_index+1);  //set up  all active layers

  // set flag to indicate that we have a mirror layer inserted
  m_have_mirror = 1;
}

// removes a mirror layer for specified layer index
void greedy_layerwise_autoencoder::remove_mirror(uint32_t layer_index) {
  if (m_have_mirror) {
    // compute layer index for mirrror
    size_t mirror_index = layer_index + 2;

    // drop the mirror layer from the model
    remove(mirror_index); ///any delete on heap, vector resize?

    // call base model setup again to reindex and set appropriate fp and bp input
    if (m_comm->am_world_master()) {
      std::cout << "Phase [" << layer_index << "] Done, Reset Layers " << std::endl;
      for(auto& l:m_layers) {
        std::cout << "Layer [ " << l->get_index() << "] #NumNeurons: " << l->get_num_neurons() << std::endl;
      }
    }
    setup();

    // set flag to indicate we've deleted our mirror layer
    m_have_mirror = 0;
  }
}


//layer wise training ends at reconstruction layer
//@todo Rewrite to get and copy all reconstruction indices to a vector(queue)
void greedy_layerwise_autoencoder::set_end_index() {
  for (size_t l =m_start_index+1; l < m_layers.size(); l++) {
    if(m_layers[l]->get_type() == "reconstruction") {
      m_end_index = l;
     return;
   }
  }
}


void greedy_layerwise_autoencoder::train(int num_epochs) {
  while(m_end_index < m_layers.size()-1) {
    set_end_index();
    train_phase(num_epochs);
    m_start_index = m_end_index; 
    m_reconstruction_layers.insert(m_reconstruction_layers.begin(),m_layers[m_end_index]);
    // move on to the next (layerwise) phase
    ++m_current_phase;
  }
  //evaluate and save all layers for i.e., (1) global cost (2) image reconstruction
  evaluate(execution_mode::testing);
  reset_phase();
}

//@todo: rename to train layer wise
void greedy_layerwise_autoencoder::train_phase(int num_epochs) {
  do_train_begin_cbs();

  // Epoch main loop
  while (get_cur_epoch() < num_epochs) {
    // Check if training has been terminated
    if (get_terminate_training()) {
      break;
    }

    // due to restart, may not always be at start of epoch
    // use mini batch index in data reader to signify start of epoch
    if (at_epoch_start()) {
      ++m_current_epoch;
      do_epoch_begin_cbs(); // needed for selected callback e.g., dump matrices
    }

    //Overide default print callback
    if (m_comm->am_world_master()) {
      //std::cout << "-----------------------------------------------------------" << std::endl;
      //std::cout << "Phase [" << m_current_phase  << "] Epoch [" << m_current_epoch << "]" <<  std::endl;
      std::cout << "\n Training hidden layer [" << m_current_phase+1  << "] at layer-wise epoch [" << m_current_epoch << "]" <<  std::endl;
      std::cout << "-----------------------------------------------------------" << std::endl;
    }
   
    //Print(m_layers[m_start_index]->get_activations()); 
    /// Set the execution mode to training
    m_execution_mode = execution_mode::training;
    for (size_t l =0; l < m_layers.size(); l++) {
      m_layers[l]->set_execution_mode(execution_mode::training);
    }

    // Train on mini-batches until data set is traversed
    // Note: The data reader shuffles the data after each epoch
    m_obj_fn->reset_statistics();
    for (auto&& m : m_metrics) {
      m->reset_metric();
    }
    bool finished_epoch = false;
    while (!finished_epoch) {
      finished_epoch = train_mini_batch();

      // save a checkpoint if needed
      if (need_checkpoint()) {
        checkpointShared();
      }
    } while(!finished_epoch);


    //print training reconstruction cost
    if (m_comm->am_world_master()) {
      std::cout << "Layer-wise training ";
    }
    m_layers[m_end_index]->epoch_print();


    do_epoch_end_cbs(); //needed for selected callback e.g., dump matrices

    evaluate_phase(execution_mode::validation);

    //print validation reconstruction cost
    if (m_comm->am_world_master()) {
      std::cout << "Layer-wise validation ";
    }
    m_layers[m_end_index]->epoch_print();

    // Reset execution mode back to training
    m_execution_mode = execution_mode::training;
    for (Layer *layer : m_layers) {
      layer->set_execution_mode(execution_mode::training);
    }

    // save checkpoint after epoch
    if (need_checkpoint()) {
      checkpointShared();
    }
  }

  do_train_end_cbs();
  m_current_epoch = 0; //reset epoch counter
}

//skip reconstruction layers that are not end indices
bool greedy_layerwise_autoencoder::train_mini_batch() {
  do_batch_begin_cbs();

  // Forward propagation
  do_model_forward_prop_begin_cbs();
  //@todo optimize: optimize considering input layer
  //for (size_t l = m_start_index; l <= m_end_index; ++l) {
  for (size_t l = 0; l <= m_end_index; ++l) {
    do_layer_forward_prop_begin_cbs(m_layers[l]);
    m_layers[l]->forward_prop();
    do_layer_forward_prop_end_cbs(m_layers[l]);
  }
  do_model_forward_prop_end_cbs();

  // Record and reset objective function value
  m_obj_fn->record_and_reset_value();

  ++m_current_step;

  // Backward propagation
  do_model_backward_prop_begin_cbs();
  //@todo optimize: optimize considering input layer
  for (size_t l = m_end_index+1; l-- > m_start_index;) {
    do_layer_backward_prop_begin_cbs(m_layers[l]);
    m_layers[l]->back_prop();
    do_layer_backward_prop_end_cbs(m_layers[l]);
  }
  do_model_backward_prop_end_cbs();

  /// Update (active) layers
  ///Freeze inactive layers
  for (size_t l = m_end_index; l > m_start_index; --l) {
    m_layers[l]->update();
  }
  const bool data_set_processed = m_layers[0]->update();
  

  do_batch_end_cbs();
  return data_set_processed;
}

void greedy_layerwise_autoencoder::evaluate_phase(execution_mode mode) {
  if (!is_execution_mode_valid(mode)) { return; }
  // Set the execution mode
  m_execution_mode = mode;
  for (size_t l = 0; l <= m_end_index; ++l) {
    m_layers[l]->set_execution_mode(mode);
  }

  // Evaluate on mini-batches until data set is traversed
  // Note: The data reader shuffles the data after each epoch
  m_obj_fn->reset_statistics();
  for (auto&& m : m_metrics) {
    m->reset_metric();
  }
  bool finished_epoch;
  do {
    finished_epoch = evaluate_mini_batch();
  } while(!finished_epoch);


  /*for (Layer* layer : m_layers) {
    layer->epoch_reset();
  }*/

  return;
}

bool greedy_layerwise_autoencoder::evaluate_mini_batch() {
  // forward propagation (mini-batch)
  for (size_t l = 0; l < m_layers.size(); l++) {
    m_layers[l]->forward_prop();
  }

  // Record and reset objective function value
  m_obj_fn->record_and_reset_value();
  
  //done processing a minibatch?  
  const bool data_set_processed = m_layers[0]->update();
  return data_set_processed;
}


void greedy_layerwise_autoencoder::evaluate(execution_mode mode) {
  if (!is_execution_mode_valid(mode)) { return; }
  //concatenate original layers with reconstruction layers
  //@todo add state (in(active)) to reconstruction layer
  m_layers.insert(std::end(m_layers), std::begin(m_reconstruction_layers)+1,std::end(m_reconstruction_layers));

  //Set appropriate layer indices and fp_input
  size_t mls = m_layers.size();
  size_t mrs_index = mls-m_reconstruction_layers.size()+1; //reconstruction layers start index
  for(size_t l = mrs_index; l < mls; ++l) {
    m_layers[l]->set_index(l);
  }

  //@todo loop for epochs??
  m_end_index = mls-1;
  evaluate_phase(mode);

  if (m_comm->am_world_master()) {
    std::cout << "Global (rel. to all (in + hidden) layers) testing ";
  }
  m_layers[m_end_index]->epoch_print();

  //@todo: finetune only up to the true layers skipping the reconstruction layers
  //m_layers.resize(m_layers.size()-m_reconstruction_layers.size());
  //clear m_reconstruction layers
  //m_reconstruction_layers.clear();

  return;
}

}  // namespace lbann
