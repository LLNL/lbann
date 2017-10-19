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
// lbann_model_planar .hpp .cpp - Sequential neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/model_planar.hpp"
#include "lbann/layers/io/io_layer.hpp"
#include "lbann/io/persist.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>

#include "mpi.h"

namespace lbann {

planar_model::planar_model(int mini_batch_size,
                                   lbann_comm *comm,
                                   objective_functions::objective_fn *obj_fn,
                                   optimizer_factory *optimizer_fac)
  : model(comm, mini_batch_size, obj_fn, optimizer_fac) {}

/**
planar_model::planar_model(const planar_model& other) :
  model(other) {
  // First copy over the layers.
  for (const auto& l : other.m_layers) {
    m_layers.push_back(l->copy());
  }
  // Update pointers for each layer.
  for (size_t l = 0; l < m_layers.size(); ++l) {
    m_layers[l]->set_neural_network_model(this);
    Layer* prev_layer = l > 0 ? m_layers[l-1] : nullptr;
    Layer* next_layer = l < m_layers.size() - 1 ? m_layers[l+1] : nullptr;
    m_layers[l]->setup_pointers(prev_layer, next_layer);
  }
  // Update target layer data readers.
  io_layer *input = dynamic_cast<io_layer*>(m_layers[0]);
  io_layer *target = dynamic_cast<io_layer*>(m_layers.back());
  if (input && target) {
    target->set_data_readers_from_layer(input);
  }
}

planar_model& planar_model::operator=(const planar_model& other) {
  model::operator=(other);
  m_layers.clear();
  // First copy over the layers.
  for (const auto& l : other.m_layers) {
    m_layers.push_back(l->copy());
  }
  // Update pointers for each layer.
  for (size_t l = 0; l < m_layers.size(); ++l) {
    m_layers[l]->set_neural_network_model(this);
    Layer* prev_layer = l > 0 ? m_layers[l-1] : nullptr;
    Layer* next_layer = l < m_layers.size() - 1 ? m_layers[l+1] : nullptr;
    m_layers[l]->setup_pointers(prev_layer, next_layer);
  }
  // Update target layer data readers.
  io_layer *input = dynamic_cast<io_layer*>(m_layers[0]);
  io_layer *target = dynamic_cast<io_layer*>(m_layers.back());
  if (input && target) {
    target->set_data_readers_from_layer(input);
  }
  return *this;
}
*/

planar_model::~planar_model() {
  // Free layers
  for (size_t h = 0; h < m_layers.size(); ++h) {
    std::vector<Layer*>& arow = m_layers[h];
    for(site_t c = 0; c < arow.size(); ++c){
      delete arow[c];
    }
  }
}

/**
bool planar_model::save_to_file(const string file_dir) {
  // get our directory name
  const char *dir = file_dir.c_str();

  // report how long this takes
  Timer timer;

  // start timer
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    timer.Start();
    printf("Saving parameters to %s ...\n", dir);
    fflush(stdout);
  }

  // create directory to hold files
  int mkdir_success = makedir(dir);
  if (! mkdir_success) {
    // failed to create the directory
    return false;
  }

  // write out details for each layer
  for (size_t l = 1; l < m_layers.size(); l++)
    if (!m_layers[l]->saveToFile(-1, dir)) {
      return false;
    }

  // stop timer
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    double secs = timer.Stop();
    printf("Saved parameters to %s (%f secs)\n", dir, secs);
    fflush(stdout);
  }

  return true;
}

bool planar_model::load_from_file(const string file_dir) {
  // get our directory name
  const char *dir = file_dir.c_str();

  // report how long this takes
  Timer timer;

  // start timer
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    timer.Start();
    printf("Loading parameters from %s ...\n", dir);
    fflush(stdout);
  }

  for (size_t l = 1; l < m_layers.size(); l++)
    if (!m_layers[l]->loadFromFile(-1, dir)) {
      return false;
    }

  // stop timer
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    double secs = timer.Stop();
    printf("Loaded parameters from %s (%f secs)\n", dir, secs);
    fflush(stdout);
  }

  return true;
}

bool planar_model::save_to_checkpoint(int fd, const char *filename, size_t *bytes) {
  // write number of layers (we'll check this on read)
  int layers = m_layers.size();
  int write_rc = write(fd, &layers, sizeof(int));
  if (write_rc != sizeof(int)) {
    fprintf(stderr, "ERROR: Failed to write number of layers to file `%s' (%d: %s) @ %s:%d\n",
            filename, errno, strerror(errno), __FILE__, __LINE__
           );
    fflush(stderr);
  }
  *bytes += write_rc;

  // write out details for each layer
  for (size_t l = 1; l < m_layers.size(); l++)
    if (!m_layers[l]->saveToCheckpoint(fd, filename, bytes)) {
      return false;
    }

  return true;
}

bool planar_model::load_from_checkpoint(int fd, const char *filename, size_t *bytes) {
  // read number of layers
  unsigned int file_layers;
  int read_rc = read(fd, &file_layers, sizeof(unsigned int));
  if (read_rc != sizeof(int)) {
    fprintf(stderr, "ERROR: Failed to read number of layers from file `%s' (%d: %s) @ %s:%d\n",
            filename, errno, strerror(errno), __FILE__, __LINE__
           );
    fflush(stderr);
  }
  *bytes += read_rc;

  if (file_layers != m_layers.size()) {
    // error!
  }

  for (size_t l = 1; l < m_layers.size(); l++) {
    if (! m_layers[l]->loadFromCheckpoint(fd, filename, bytes)) {
      return false;
    }
  }

  return true;
}
*/

struct lbann_model_planar_header {
  uint32_t layers;
};

bool planar_model::save_to_checkpoint_shared(persist& p) {
  // write parameters from base class first
  model::save_to_checkpoint_shared(p);

  // write a single header describing layers and sizes?

  // have rank 0 write the network file
  if (p.get_rank() == 0) {
    uint32_t layers = m_layers.size();
    p.write_uint32(persist_type::model, "layers", (uint32_t) layers);

    // TODO: record each layer type and size (to be checked when read back)
  }

  // write out details for each layer
  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->saveToCheckpointShared(p)) {
      return false;
    }
  }

  return true;
}

bool planar_model::load_from_checkpoint_shared(persist& p) {
  // read parameters from base class first
  model::load_from_checkpoint_shared(p);

  // have rank 0 read the network file
  struct lbann_model_planar_header header;
  if (p.get_rank() == 0) {
    p.read_uint32(persist_type::model, "layers", &header.layers);

    // TODO: read back each layer type and size
  }

  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (header.layers != m_layers.size()) {
    // error!
    return false;
  }

  // TODO: check that each layer type matches what we'd expect

  // read in each layer
  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->loadFromCheckpointShared(p)) {
      return false;
    }
  }

  return true;
}

/***
 * Given a new layer, create 'num_heads' copies of the new layer and add them to
 * the horizontal_index-th level of the model. */
//int planar_model::stackup_duplicate(int horizontal_index, int num_heads, Layer *new_layer){ }

/// Add the new layer at hindex-th of current level
int planar_model::stackup_tail(int hindex, Layer *new_layer){
  /// Resize layer vector to m_width if a new layer is added first time 
  if(m_layers.size() == vertical_index){
    std::vector<Layer *> new_level;
    m_layers.push_back(new_level);
  }

  /// Do some sanity check first
  if(m_layers.size() != hindex+1){
    std::cerr << "Adding new layer to invalid level of plane" << "\n";
    throw lbann::lbann_exception("Adding new layer to invalid level of plane");
  }
  if(m_layers[hindex].size() >= m_width){
    std::cerr << "Adding new layer to planar level that is already full" << "\n";
    throw lbann::lbann_exception("Adding new layer to planar level that is already full");
  }

  std::vector<Layer *>& current_level = m_layers[hindex];
  new_layer->set_index(hindex);
  current_level.push_back(new_layer);
  return hindex;
}

/// Add multiple (m_width) copies of the new layer at the next level
int planar_model::stackup_duplicate(Layer *new_layer){
  /// A new level of m_width layers will be created
  std::vector<Layer *> new_level;

  new_level.push_back(new_layer);
  for(uint j=1; j<m_width; j++){
    Layer layer_copy = *new_layer;
    new_level.push_back(layer_copy); 
  }
  m_layers.push_back(new_level);

  return m_layers.size();
}

/**
void planar_model::remove(int index) {
  delete m_layers[index];
  m_layers.erase(m_layers.begin()+index);
}

void planar_model::insert(int index, Layer *new_layer) {
  m_layers.insert(m_layers.begin()+index, new_layer);
}

Layer *planar_model::swap(int index, Layer *new_layer) {
  Layer *tmp = m_layers[index];
  m_layers[index] = new_layer;
  return tmp;
} */

void planar_model::setup(int start_index, int end_index) {
  if(end_index <= 0) {
    end_index = m_layers.size();
  }

  /// Setup each layer
  Layer* prev_layer;
  Layer* next_layer;
  for (int l=start_index; l<end_index; l++) {
    std::vector<Layers *>& horizontal_layers = m_layers[l];
    for(int j=0; j<m_layers[l].size(); j++) {
      /// Determine the previous layer
      if(l <= 0){
        prev_layer = nullptr;
      }
      else{
        if(m_layers[l-1].size() < m_layers[l].size()){/// Fan-out structure
          assert(m_layers[l-1].size() == 1);
          prev_layer = m_layers[l-1].at(0);
        }
        else if(m_layers[l-1].size() > m_layers[l].size()){/// Fan-in structure
          assert(m_layers[l].size() == 1);
          prev_layer = m_layers[l-1].at(j);
        }
        else{/// Current and previous layers have the same number of layers
          prev_layer = m_layers[l-1].at(j);
        }
      }

      /// Determine the next layer
      if(l >= end_index-1){
        next_layer = nullptr;
      }
      else{
        if(m_layers[l+1].size() < m_layers[l].size()){/// Fain-in structure
          assert(m_layers[l+1].size() == 1);
          next_layer = m_layers[l+1].at(0);
        }
        else if(m_layers[l+1].size() > m_layers[l].size()){/// Fan-out structure
          assert(m_layers[l].size() == 1);
          next_layer = m_layers[l+1].at(j);
        }
        else{//// Current and the next layer has the same number of layers
          next_layer = m_layers[l+1].at(j);
        }
      }
    
      
      Layer* current_layer = horizontal_layer[j];
      current_layer->set_neural_network_model(this); /// Provide a reverse point from each layer to the model
      current_layer->setup(prev_layer, next_layer);
      current_layer->check_setup();
      current_layer->set_index(l);
      if (m_comm->am_world_master()) {
        cout << std::setw(3) << l << ":[" << std::setw(18) << current_layer->get_type() <<  "] Set up a layer with input " << std::setw(7) << current_layer->get_num_prev_neurons() << " and " << std::setw(7) << current_layer->get_num_neurons() << " neurons."  << endl;
      }
    }
  }

  /// Share the weights between Siamese heads
  equalize(start_index, end_index);

  // Set up callbacks
  /// XXXXXXXXXXXX
  /// Following needs to be changed to accomodate this planar model
  setup_callbacks();
}

/** We are ignoring callbacks at this moment. Currently all callback routines assume 
 * sequential model as their baseline model, which will not work with the new
 * planar model. This issue will be addressed in the future. */

void planar_model::train(int num_epochs) {
  /// Igroring callback
  // do_train_begin_cbs();

  // Epoch main loop
  for (int epoch = 0; epoch < num_epochs; ++epoch) {

    // Check if training has been terminated
    if (get_terminate_training()) {
      break;
    }

    // due to restart, may not always be at start of epoch
    // use mini batch index in data reader to signify start of epoch
    if (at_epoch_start()) {
      ++m_current_epoch;
      /// Igroring callback
      // do_epoch_begin_cbs();
    }

    /// Set the execution mode to training
    m_execution_mode = execution_mode::training;
    for (size_t l = 0; l < m_layers.size(); ++l) {
      std::vector<Layers *>& horizontal_layer = m_layers[l];
      for(size_t j=0; j<horizontal_layer.size(); j++) {
        horizontal_layer[j]->set_execution_mode(execution_mode::training);
      }
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
    }

    // Evaluate model on validation set
    // TODO: do we need validation callbacks here?
    // do_validation_begin_cbs();
    evaluate(execution_mode::validation);
    // do_validation_end_cbs();

    /// Igroring callback
    // do_epoch_end_cbs();

    // save checkpoint after epoch
    if (need_checkpoint()) {
      checkpointShared();
    }
  }

  /// Igroring callback
  // do_train_end_cbs();
}

bool planar_model::train_mini_batch() {
  /// Igroring callback
  // do_batch_begin_cbs();

  /// Igroring callback
  // Forward propagation
  // do_model_forward_prop_begin_cbs();
  for (size_t l = 0u; l < m_layers.size(); ++l) {
    std::vector<Layers *>& horizontal_layer = m_layers[l];
    for(size_t j=0; j<horizontal_layer.size(); j++) {
      /// Igroring callback
      // do_layer_forward_prop_begin_cbs(horizontal_layer[j]);
      horizontal_layer[j]->forward_prop();
      /// Igroring callback
      // do_layer_forward_prop_end_cbs(hosrizontal_layer[j]);
    }
  }
  /// Igroring callback
  // do_model_forward_prop_end_cbs();

  // Record and reset objective function value
  m_obj_fn->record_and_reset_value();

  /// Igroring callback
  // Backward propagation
  // do_model_backward_prop_begin_cbs();
  for (size_t l = m_layers.size(); l-- > 0u;) {
    std::vector<Layers *>& horizontal_layer = m_layers[l];
    for(size_t j=0; j<horizontal_layer.size(); j++) {
      /// Igroring callback
      // do_layer_backward_prop_begin_cbs(m_layers[l]);
      hosrizontal_layer[j]->back_prop();
      /// Igroring callback
      // do_layer_backward_prop_end_cbs(m_layers[l]);
    }
  }
  /// Igroring callback
  // do_model_backward_prop_end_cbs();

  /// Sums up gradients before update so that the weights at multi-headed level are
  // 'tied' together.
  // XXXX: TO DO; how to update using new gradients
  sum_up_gradients();

  /// Update layers
  for (size_t l = m_layers.size() - 1; l > 0u; --l) {
    std::vector<Layers *>& horizontal_layer = m_layers[l];
    for(size_t j=0; j<horizontal_layer.size(); j++) {
      horizontal_layer[j]->update();
    }
  }
  /// Ensure the first level of the planar model consists of single layer.
  assert(m_layers[0].size() == 1);
  const bool data_set_processed = m_layers[0].at(0)->update();

  /// Igroring callback
  // do_batch_end_cbs();
  ++m_current_step; // Update the current step once the entire mini-batch is complete
  return data_set_processed;
}

bool planar_model::at_epoch_start() {
  // use mini batch index in data reader to signify start of epoch
  io_layer *input = (io_layer *) m_layers[0];
  bool flag = input->at_new_epoch();
  return flag;
}

void planar_model::equalize(int  start_index, int end_index)
{
  for (int l=start_index; l<end_index; l++) {
    /// No need to copy weights for single-head level
    if(m_layers[l].size() <= 1)
      continue;

    /// Make sure all layers at current level are not a mix of learning and non-learning layers
    char same_type = (m_layers[l].at(0)->is_learning_layer()) ? 1 : 0;
    for(size_t k=1; k<m_layers[l].size(); k++){
      same_type ^= (m_layers[l].at(k)->is_learning_layer()) ? 1 : 0;
    }
    assert(!same_type);

    // All layers at current level are non-learning layers, so skip
    if(!m_layers[l].at(0)->is_learning_layer())
      continue;

    /// Copy weights between heads
    /// In case when only weights are shared
    ElMat& anchor_weights = m_layers[l].at(0)->get_weights();

    for(size_t k=1; k<m_layers[l].size(); k++){
      ElMat& target_weights = m_layers[l].at(k)->get_weights();
      Copy(anchor_weights, target_weights);
    }
  }
}


void planar_model::sum_up_gradients()
{
  for(size_t l=0; l<m_layers.size(); l++){
    /// No need to copy weights for this layer
    if(m_layers[l].size() <= 1)
      continue;
  
    /// Make sure all layers at current level are not a mix of learning and non-learning layers
    char same_type = (m_layers[l].at(0)->is_learning_layer()) ? 1 : 0;
    for(size_t k=1; k<m_layers[l].size(); k++){
      same_type ^= (m_layers[l].at(k)->is_learning_layer()) ? 1 : 0;
    }
    assert(!same_type);

    // All layers at current level are non-learning layers, so skip
    if(!m_layers[l].at(0)->is_learning_layer())
      continue;

    /// Sum up weights_gradient from each layer at current level
    ElMat& weights_gradient_sum = m_layers[l].at(0)->get_weights_gradient();
    for(size_t k=1; k<m_layers[l].size(); k++){
      ElMat& current_gradient = m_layers[l].at(k)->get_weights_gradient();
      weights_gradient_sum += current_gradient;
    }
    for(size_t k=0; k<m_layers[l].size(); k++){
      ElMat& current_gradient = m_layers[l].at(k)->get_weights_gradient();
      current_gradient = weights_gradient_sum;
    }
  }
}

}  // namespace lbann


void planar_model::evaluate(execution_mode mode) {
  if (!is_execution_mode_valid(mode)) { return; }
  switch(mode) {
  case execution_mode::validation:
    /// Igonoring callbacks for now
    //do_validation_begin_cbs();
    break;
  case execution_mode::testing:
    /// Igonoring callbacks for now
    //do_test_begin_cbs();
    break;
  default:
    throw lbann_exception("Illegal execution mode in evaluate function");
  }

  // Set the execution mode for each layer 
  m_execution_mode = mode;
  for (size_t l = 0u; l < m_layers.size(); ++l) {
    for(size-t k = 0u; k < m_layers[l].size(); k++) {
      m_layers[l].at(k)->set_execution_mode(mode);
    }
  }

  // Evaluate on mini-batches until data set is traversed
  // Note: The data reader shuffles the data after each epoch
  m_obj_fn->reset_statistics();
  for (auto&& m : m_metrics) {
    m->reset_metric();
  }
  bool finished_epoch = false;
  while (!finished_epoch) {
    finished_epoch = evaluate_mini_batch();
  }

  switch(mode) {
  case execution_mode::validation:
    /// Igonoring callbacks for now
    //do_validation_end_cbs();
    break;
  case execution_mode::testing:
    /// Igonoring callbacks for now
    //do_test_end_cbs();
    break;
  default:
    throw lbann_exception("Illegal execution mode in evaluate function");
  }

  return;
}

bool planar_model::evaluate_mini_batch() {
  /// Igroring callback
  // do_batch_evaluate_begin_cbs();

  /// Igroring callback
  // forward propagation (mini-batch)
  // do_model_evaluate_forward_prop_begin_cbs();
  for (size_t l = 0u; l < m_layers.size(); l++) {
    std::vector<Layers *>& horizontal_layer = m_layers[l];
    for(size_t j=0; j<horizontal_layer.size(); j++) {
      /// Igroring callback
      // do_layer_evaluate_forward_prop_begin_cbs(m_layers[l]);
      horizontal_layer[j]->forward_prop();
      /// Igroring callback
      // do_layer_evaluate_forward_prop_end_cbs(m_layers[l]);
    }
  }
  /// Igroring callback
  // do_model_evaluate_forward_prop_end_cbs();

  // Record and reset objective function value
  m_obj_fn->record_and_reset_value();

  // Update layers
  // Note: should only affect the input and target layers
  for (size_t l = m_layers.size() - 1; l > 0u; --l) {
    std::vector<Layers *>& horizontal_layer = m_layers[l];
    for(size_t j=0; j<horizontal_layer.size(); j++) {
      horizontal_layer[j]->update();
    }
  }
  /// Ensure the first level of the planar model consists of single layer.
  assert(m_layers[0].size() == 1);
  const bool data_set_processed = m_layers[0].at(0)->update();

  /// Igroring callback
  // do_batch_evaluate_end_cbs();
  switch(m_execution_mode) {
  case execution_mode::validation:
    ++m_current_validation_step;
    break;
  case execution_mode::testing:
    ++m_current_testing_step;
    break;
  default:
    throw lbann_exception("Illegal execution mode in evaluate mini-batch function");
  }
  return data_set_processed;
}
