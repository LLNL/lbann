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
// sequential .hpp .cpp - Sequential neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/sequential.hpp"
#include <unordered_set>

namespace lbann {

sequential_model::sequential_model(lbann_comm *comm,
                                   int mini_batch_size,
                                   objective_function *obj_fn,
                                   optimizer* default_optimizer)
  : model(comm, mini_batch_size, obj_fn, default_optimizer) {}

void sequential_model::setup_layer_topology() {

  // Set up parent/child relationships between adjacent layers
  for (size_t i = 1; i < m_layers.size(); ++i) {
    m_layers[i]->add_parent_layer(m_layers[i-1]);
  }
  for (size_t i = 0; i < m_layers.size() - 1; ++i) {
    m_layers[i]->add_child_layer(m_layers[i+1]);
  }

  // Setup layer graph
  model::setup_layer_topology();

  // Make sure that execution order is valid
  std::set<int> nodes;
  std::map<int,std::set<int>> edges;
  construct_layer_graph(nodes, edges);
  if (!graph::is_topologically_sorted(nodes, edges)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "layer execution order is not topologically sorted";
    throw lbann_exception(err.str());
  }

  freeze_layers_under_frozen_surface();
}

bool sequential_model::save_to_checkpoint(int fd, const char *filename, size_t *bytes) {
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
  /*for (size_t l = 1; l < m_layers.size(); l++)
    if (!m_layers[l]->saveToCheckpoint(fd, filename, bytes)) {
      return false;
    }*/

  return true;
}

bool sequential_model::load_from_checkpoint(int fd, const char *filename, size_t *bytes) {
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

  /*for (size_t l = 1; l < m_layers.size(); l++) {
    if (! m_layers[l]->loadFromCheckpoint(fd, filename, bytes)) {
      return false;
    }
  }*/

  return true;
}

struct lbann_model_sequential_header {
  uint32_t layers;
};

bool sequential_model::save_to_checkpoint_shared(persist& p, bool val_end) {
  // write parameters from base class first
  model::save_to_checkpoint_shared(p, val_end);

  // write a single header describing layers and sizes?

  // have rank 0 write the network file
  if (p.get_rank() == 0 && !val_end) {
    uint32_t layers = m_layers.size();
    p.write_uint32(persist_type::model, "layers", (uint32_t) layers);

    // TODO: record each layer type and size (to be checked when read back)
  }
  // write out details for each layer
  if(!val_end){
    for (weights *w : m_weights) {
      w->save_to_checkpoint_shared(p);
    }
      for (size_t l = 0; l < m_layers.size(); l++) {
      if (! m_layers[l]->save_to_checkpoint_shared(p,val_end)) {
        return false;
      }
  }
  }
  else if(val_end){
    for (size_t l = 0; l < m_layers.size(); l++) {
      if (! m_layers[l]->save_to_checkpoint_shared(p,val_end)) {
        return false;
      }
    }
  }
  //m_objective_function->save_to_checkpoint_shared(p);
  return true;
}

void sequential_model::write_proto(lbann_data::Model* proto) {

  model::write_proto(proto);
  //Add layers
  if (m_comm->am_world_master()) {
    proto->set_name(name());
    for(size_t l = 0; l < m_layers.size(); l++) {
      auto layer_proto = proto->add_layer();
      m_layers[l]->write_proto(layer_proto);
    }
  }
}

bool sequential_model::load_from_checkpoint_shared(persist& p) {
  // read parameters from base class first
  model::load_from_checkpoint_shared(p);

  // have rank 0 read the network file
  struct lbann_model_sequential_header header;
  if (p.get_rank() == 0) {
    p.read_uint32(persist_type::model, "layers", &header.layers);

    // TODO: read back each layer type and size
  }

  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (header.layers != m_layers.size()) {
    // error!
     std::stringstream err;
     err << __FILE__ << " " << __LINE__ << " :: "
         << "Error occured in model reload: model layers not equal";
     throw lbann_exception(err.str());
     return false;
  }

  // TODO: check that each layer type matches what we'd expect
  for (weights *w : m_weights) {
    w->load_from_checkpoint_shared(p);
  }
  // read in each layer
  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->load_from_checkpoint_shared(p)) {
      return false;
    }
  }
  //m_objective_function->load_from_checkpoint_shared(p);
  return true;
}

}  // namespace lbann
