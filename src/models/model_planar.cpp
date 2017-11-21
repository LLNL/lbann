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
// model_planar .hpp .cpp - Planar neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/model_planar.hpp"
#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/layers/io/target/target_layer.hpp"

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
                                   objective_functions::objective_function *obj_fn,
                                   optimizer_factory *optimizer_fac,
                                   int width)
  : model(comm, mini_batch_size, obj_fn, optimizer_fac), m_width(width)
{}

planar_model::planar_model(const planar_model& other)
  : model(other), m_width(other.m_width), m_head_counts(other.m_head_counts) {
  // First copy over the layers.
  copy_layers(other.m_layers);
}

planar_model& planar_model::operator=(const planar_model& other) {
  model::operator=(other);
  m_width = other.m_width;
  m_head_counts = other.m_head_counts;
  // First copy over the layers.
  copy_layers(other.m_layers);
  return *this;
}

planar_model::~planar_model() {
  delete_layers();
}

void planar_model::delete_layers() {
  for (auto& layer_peers : m_layers) {
    for (auto layer : layer_peers) {
      delete layer;
    }
  }
  m_layers.clear();
}

void planar_model::copy_layers(const Layer_stack_t& src_stack) {
  delete_layers();
  Layer_map_t map_src_to_new;

  for (const auto& src_peers : src_stack) {
    m_layers.push_back(Layer_peers_t());
    auto& new_peers = m_layers.back();
    for (const auto& src_layer : src_peers) {
      try {
        Layer* new_layer = src_layer->copy();
        map_src_to_new[src_layer] = new_layer;
        new_peers.push_back(new_layer);
      } catch (std::bad_alloc&) {
        throw("Planar model: Failed to copy a layer");
      }
    }
  }
  renew_layer_links(src_stack, map_src_to_new);
}

void planar_model::set_layers(const Layer_stack_t& new_stack) {
  delete_layers();
  m_layers = new_stack;
}

void planar_model::renew_layer_links(const Layer_stack_t& src_stack,
                                     const Layer_map_t& map_src_to_new) const {
  for (auto&& src_peers : src_stack) {
    for (auto&& src_layer : src_peers) {
      Layer* new_layer = find_layer(map_src_to_new, src_layer);
      std::vector<Layer *> src_pointers = src_layer->get_layer_pointers();
      std::vector<Layer *> new_pointers;
      for (const Layer* src_pointer : src_pointers) {
        Layer* new_pointer = find_layer(map_src_to_new, src_pointer);
        new_pointers.push_back(new_pointer);
      }
      new_layer->set_layer_pointers(new_pointers);
    }
  }
}

Layer* planar_model::find_layer(const Layer_map_t& map_src_to_new, const Layer* const src_layer) {
  //Layer_map_t::const_iterator it = map_src_to_new.find(src_layer);
  Layer_map_t::const_iterator it = map_src_to_new.end();
  if (it == map_src_to_new.end()) return nullptr;
  return it->second;
}

void planar_model::add(Layer *layer){
  if (layer == nullptr) {
    throw lbann_exception("Planar model: Attempted to add null pointer as a layer.");
  }

  // Add layer to a new layer set
  m_layers.push_back(Layer_peers_t());
  Layer_peers_t& new_layer_peers = m_layers.back();
  new_layer_peers.push_back(layer);
}

/***
 * Given a new layer, create 'K' copies of the new layer and add them to
 * the next level. */
void planar_model::stackup_duplicate(Layer_peers_t& layer_peers, int num_heads) {
  /// A new level of num_heads layers will be created
  if (num_heads <= 0) {
    // allready verified that ((layer_peers.size() == 1u) && layer_peers[0]) in the calling context
    throw lbann_exception("Planar model: layer level does not have any layer to copy.");
  }
  const auto master_layer = layer_peers.at(0);
  /* The following condition is already checked in the calling context
  if ((master_layer->is_fanin_layer() || master_layer->is_fanout_layer())
      && (num_heads > 1u)) {
    throw lbann_exception("Planar model: fanin/fanout layer must not be duplicated.");
  }
  */
  layer_peers.reserve(num_heads);
  const std::string layer_name = master_layer->get_name();
  master_layer->set_name("h1_" + layer_name);

  for(int k=1; k < num_heads; k++){
    try {
      auto layer_copy = master_layer->copy();
      layer_copy->set_name("h" + std::to_string(k+1) + "_" + layer_name);
      layer_peers.push_back(layer_copy);
    } catch (std::bad_alloc&) {
      throw("Planar model: Failed to duplicate a layer");
    }
  }
}

void planar_model::setup() {
  bool multi_headed = false;

  /// Convert sequential layers to planar layers
  for(size_t l=0; l<m_layers.size(); l++){
    assert(m_layers[l].size() == 1);

    Layer *layer = m_layers[l].at(0);

    if(!multi_headed){
      /// Currently in single-head state

      if(layer->is_fanin_layer()){
        /// Cannot fan in from single-head state
        std::cerr << "Cannot fan in from single-head state" << "\n";
        throw lbann::lbann_exception("Cannot fan in from single-head state");
      } else if(layer->is_fanout_layer()) {
        /// Fanning out layers to multi-head state
        multi_headed = true;
        //stackup_duplicate(layer, 1);
      } else{
        /// layer is already in m_layers; no action is required
        //stackup_duplicate(layer, 1);
      }
    } else{
      /// Currently in multi head state
      if(layer->is_fanout_layer()){
        /// Cannot fan out from multi-head state
        std::cerr << "Cannot fan out from multi-head state" << "\n";
        throw lbann::lbann_exception("Cannot fan out from multi-head state");
      } else if(layer->is_fanin_layer()){
        /// Fanning in from multi-head state; no action is needed
        //stackup_duplicate(layer, 1);
        multi_headed = false;
      } else{
        /// Expand current layer to m_width heads
        const std::string layer_name = layer->get_name();
        layer->set_name("h1_" + layer_name);
        for(int k=1; k<m_width; k++){
          Layer *layer_copy = layer->copy();
          layer_copy->set_name("h" + std::to_string(k+1) + "_" + layer_name);
          m_layers[l].push_back(layer_copy);
        }
        //stackup_duplicate(layer, m_width);
      }
    }
  }
  setup_subset();
}

void planar_model::setup_subset() {

  for (size_t l=0u; l<m_layers.size(); ++l) {
    std::vector<Layer *>& horizontal_layers = m_layers[l];

    for(size_t k=0u; k<horizontal_layers.size(); ++k) {

      Layer* current_layer = horizontal_layers[k];

      // Provide a reverse point from each layer to the model
      current_layer->set_neural_network_model(this);
      // setup links to parent layers
      if (l <= 0u) {
        current_layer->add_parent_layer(nullptr);
      } else {
        for(size_t i=0u; i < m_layers[l-1].size(); ++i)
          current_layer->add_parent_layer(m_layers[l-1][i]);
      }
      // setup links to children layers
      if (l+1 >= m_layers.size()) {
        current_layer->add_child_layer(nullptr);
      } else {
        for(size_t i=0u; i < m_layers[l+1].size(); ++i)
          current_layer->add_child_layer(m_layers[l+1][i]);
      }

      current_layer->setup();
      current_layer->check_setup();

      if (m_comm->am_world_master()) {
        std::cout << print_layer_description(current_layer) << std::endl;
      }
    }
    if (!check_layer_type_consistency(horizontal_layers)) {
      throw("Planar model: layer type consistency failed");
    }
  }
  /// Share the weights between Siamese heads
  equalize();

  // Set up callbacks
  /// XXXXXXXXXXXX
  /// Following needs to be changed to accomodate this planar model
  setup_callbacks();
}

/** Make sure all layers at current level are not a mix of learning and
 *  non-learning layers nor of optimizable and non-optimizable layers
 */
bool planar_model::check_layer_type_consistency(const Layer_peers_t& layer_peers) const {
  /// No need to check for single-head level
  if (layer_peers.size() <= 1u)
    return true;

  const bool optimizable_type
    = (dynamic_cast<const optimizable_layer*>(layer_peers[0]) != nullptr);
  if (optimizable_type) {
    for (auto&& layer : layer_peers) {
      const optimizable_layer* olayer
        = dynamic_cast<const optimizable_layer*>(layer);
      if (olayer == nullptr) return false;
    }
  } else
    return true;

  const bool learning_type
    = (dynamic_cast<const learning*>(layer_peers[0]) != nullptr);
  if (learning_type) {
    for (auto&& layer : layer_peers) {
      const learning* llayer = dynamic_cast<const learning*>(layer);
      if (llayer == nullptr) return false;
    }
  }
  return true;
}


////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

/// Forward propagation in planar model with callbacks for layer evaluation
void planar_model::forward_prop_to_evaluate() {
  // Forward propagation
  do_model_evaluate_forward_prop_begin_cbs();
  for (const auto& layer_peers : m_layers) {
    for (auto const layer : layer_peers) {
      do_layer_evaluate_forward_prop_begin_cbs(layer);
      layer->forward_prop();
      do_layer_evaluate_forward_prop_end_cbs(layer);
    }
  }
  do_model_evaluate_forward_prop_end_cbs();
}

/// Update target and input layers
bool planar_model::update_io_layers() {
  bool finished = true;
  for (const auto& layer_peers : m_layers) {
    for (auto const layer : layer_peers) {
      target_layer* const target = dynamic_cast<target_layer *>(layer);
      if (target != nullptr) {
        target->update();
      }
    }
  }
  for (const auto& layer_peers : m_layers) {
    for (auto const layer : layer_peers) {
      input_layer* const input = dynamic_cast<input_layer *>(layer);
      if (input != nullptr) {
        finished = input->update() && finished;
      }
    }
  }
  return finished;
}

/// Forward propagation in planar model
void planar_model::forward_prop() {
  do_model_forward_prop_begin_cbs();
  for (const auto& layer_peers : m_layers) {
    for (auto const layer : layer_peers) {
      do_layer_forward_prop_begin_cbs(layer);
      layer->forward_prop();
      do_layer_forward_prop_end_cbs(layer);
    }
  }
  do_model_forward_prop_end_cbs();
}

/// Backward propagation in planar model
void planar_model::backward_prop() {
  do_model_backward_prop_begin_cbs();
  for (size_t p = m_layers.size(); p-- > 0u;) {
    for (size_t l = m_layers[p].size(); l-- > 0u; ) {
      Layer* const layer = m_layers[p][l];
      do_layer_backward_prop_begin_cbs(layer);
      layer->back_prop();
      do_layer_backward_prop_end_cbs(layer);
    }
  }
  do_model_backward_prop_end_cbs();
}

/// equalize non-master layers' weights with master layer's parameters
void planar_model::equalize() {
  for (const auto& layer_peers : m_layers) {
    if (layer_peers.size() < 2u) continue;
    Layer* const master_layer = layer_peers[0];
    optimizable_layer* const master_opt_layer
      = dynamic_cast<optimizable_layer*>(master_layer);
    if (master_opt_layer == nullptr) continue;

    const AbsDistMat& parameters = master_opt_layer->get_parameters();
    for (Layer* const layer : layer_peers) {
      if (layer == master_layer) continue;
      optimizable_layer* const opt_layer
        = dynamic_cast<optimizable_layer*>(layer);
      opt_layer->clear_parameters_gradient();
      opt_layer->set_parameters(parameters);
    }
  }
}

void planar_model::update_optimizable_layers() {
  // Update optimizable layers
  // Note: We iterate through layer groups that are comprised of
  // optimizable layers which is flaged to share weights.
  for (const auto& layer_peers : m_layers) {
    Layer* const master_layer = layer_peers[0];
    optimizable_layer* const master_opt_layer
      = dynamic_cast<optimizable_layer*>(master_layer);
    if (master_opt_layer == nullptr) continue;

    // Accumulate gradients in master layer
    for (Layer* const layer : layer_peers) {
      if (layer == master_layer) continue;
      optimizable_layer* const opt_layer
        = dynamic_cast<optimizable_layer*>(layer);
      const AbsDistMat& gradient = opt_layer->get_parameters_gradient();
      master_opt_layer->add_to_parameters_gradient(gradient);
      opt_layer->clear_parameters_gradient();
    }

    // Update parameters in master layer
    master_layer->update();
    master_opt_layer->clear_parameters_gradient();

    // Update non-master layers with master layer's parameters
    const AbsDistMat& parameters = master_opt_layer->get_parameters();
    for (Layer* const layer : layer_peers) {
      if (layer == master_layer) continue;
      optimizable_layer* const opt_layer
        = dynamic_cast<optimizable_layer*>(layer);
      opt_layer->set_parameters(parameters);
    }
  }
}

void planar_model::set_execution_mode(execution_mode mode) {
  m_execution_mode = mode;
  const Layer_stack_t& layer_stack = get_layers();
  for (auto&& layer_peers : layer_stack) {
    for (auto&& layer : layer_peers) {
      layer->set_execution_mode(mode);
    }
  }
}

bool planar_model::is_execution_mode_valid(execution_mode mode) const {
  const Layer_stack_t& layer_stack = get_layers();
  for (auto&& layer_peers : layer_stack) {
    for (auto&& layer : layer_peers) {
      const input_layer* const input = dynamic_cast<const input_layer*>(layer);
      if (input != nullptr && !(input->is_execution_mode_valid(mode))) {
        return false;
      }
    }
  }
  return true;
}

////////////////////////////////////////////////////////////
// Summarizer
////////////////////////////////////////////////////////////

void planar_model::summarize_stats(lbann_summary& summarizer) {
  const Layer_stack_t& layer_stack = get_layers();
  for (auto&& layer_peers : layer_stack) {
    for (auto&& layer : layer_peers) {
      layer->summarize_stats(summarizer, get_cur_step());
    }
  }
}

void planar_model::summarize_matrices(lbann_summary& summarizer) {
  const Layer_stack_t& layer_stack = get_layers();
  for (auto&& layer_peers : layer_stack) {
    for (auto&& layer : layer_peers) {
      layer->summarize_matrices(summarizer, get_cur_step());
    }
  }
}

////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

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

/**
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
    for(size_t k=0; k< m_layers[l].size(); k++){
      Layer *current_layer = m_layers[l].at(k);
      if (! current_layer->saveToCheckpointShared(p)) {
        return false;
      }
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
    for(size_t k=0; k<m_layers[l].size(); k++){
      Layer *current_layer = m_layers[l].at(k);
      if (! current_layer->loadFromCheckpointShared(p)) {
        return false;
      }
    }
  }

  return true;
}
*/


}  // namespace lbann
