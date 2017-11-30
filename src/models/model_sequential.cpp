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
// lbann_model_sequential .hpp .cpp - Sequential neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/model_sequential.hpp"
#include "lbann/layers/io/io_layer.hpp"
#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/layers/io/target/target_layer.hpp"
#include "lbann/io/persist.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "mpi.h"

namespace lbann {

sequential_model::sequential_model(lbann_comm *comm,
                                   int mini_batch_size,
                                   objective_function *obj_fn,
                                   optimizer* default_optimizer)
  : model(comm, mini_batch_size, obj_fn, default_optimizer) {}

void sequential_model::remove(int index) {
  if (m_layers[index]) {
    delete m_layers[index];
  }
  m_layers.erase(m_layers.begin() + index);
}

void sequential_model::insert(int index, Layer *layer) {
  m_layers.insert(m_layers.begin() + index, layer);
}

Layer *sequential_model::swap(int index, Layer *layer) {
  Layer *tmp = m_layers[index];
  m_layers[index] = layer;
  return tmp;
}

void sequential_model::setup() {
  setup_subset(0, m_layers.size());
}

void sequential_model::setup_subset(int start_index, int end_index) {

  // Setup each layer
  for (int l=start_index; l<end_index; ++l) {
    m_layers[l]->set_neural_network_model(this); /// Provide a reverse point from each layer to the model
    if (l > 0) {
      m_layers[l]->add_parent_layer(m_layers[l-1]);
    }
    if (l < end_index - 1) {
      m_layers[l]->add_child_layer(m_layers[l+1]);
    }
    m_layers[l]->setup();
    m_layers[l]->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << print_layer_description(m_layers[l]) << std::endl;
    }
  }

  // Setup objective function
  m_objective_function->setup(*this);

  // Set up callbacks
  setup_callbacks();
}

int sequential_model::num_previous_neurons() {
  if (m_layers.size() == 0) {
    return -1;
  }
  Layer *prev_layer = m_layers.back();
  return prev_layer->get_num_neurons();
}

#if 0

bool sequential_model::save_to_file(const string file_dir) {
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

bool sequential_model::load_from_file(const string file_dir) {
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
#endif
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

bool sequential_model::save_to_checkpoint_shared(persist& p) {
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

  for (weights *w : m_weights) {
    w->saveToCheckpointShared(p);
  }

  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->saveToCheckpointShared(p)) {
      return false;
    }
  }
  //m_objective_function->saveToCheckpointShared(p);
  return true;
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
    return false;
  }

  // TODO: check that each layer type matches what we'd expect
  for (weights *w : m_weights) {
    w->loadFromCheckpointShared(p);
  }
  // read in each layer
  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->loadFromCheckpointShared(p)) {
      return false;
    }
  }
  //m_objective_function->loadFromCheckpointShared(p);
  return true;
}

}  // namespace lbann
