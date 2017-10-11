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
#include "lbann/io/persist.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>

#include "mpi.h"

namespace lbann {

sequential_model::sequential_model(int mini_batch_size,
                                   lbann_comm *comm,
                                   objective_functions::objective_function *obj_fn,
                                   optimizer_factory *optimizer_fac)
  : model(comm, mini_batch_size, obj_fn, optimizer_fac) {}

sequential_model::sequential_model(const sequential_model& other) :
  model(other) {
  // First copy over the layers.
  for (const auto& l : other.m_layers) {
    m_layers.push_back(l->copy());
  }
  // Update pointers for each layer.
  for (size_t l = 0; l < m_layers.size(); ++l) {
    m_layers[l]->set_neural_network_model(this);
    if (l > 0) {
      m_layers[l]->get_parent_layers().front() = m_layers[l-1];
    }
    if (l < m_layers.size() - 1) {
      m_layers[l]->get_child_layers().front() = m_layers[l+1];
    }
  }
  // Update target layer data readers.
  io_layer *input = dynamic_cast<io_layer*>(m_layers[0]);
  io_layer *target = dynamic_cast<io_layer*>(m_layers.back());
  if (input && target) {
    target->set_data_readers_from_layer(input);
  }
}

sequential_model& sequential_model::operator=(const sequential_model& other) {
  model::operator=(other);
  m_layers.clear();
  // First copy over the layers.
  for (const auto& l : other.m_layers) {
    m_layers.push_back(l->copy());
  }
  // Update pointers for each layer.
  for (size_t l = 0; l < m_layers.size(); ++l) {
    m_layers[l]->set_neural_network_model(this);
    if (l > 0) {
      m_layers[l]->get_parent_layers().front() = m_layers[l-1];
    }
    if (l < m_layers.size() - 1) {
      m_layers[l]->get_child_layers().front() = m_layers[l+1];
    }
  }
  // Update target layer data readers.
  io_layer *input = dynamic_cast<io_layer*>(m_layers[0]);
  io_layer *target = dynamic_cast<io_layer*>(m_layers.back());
  if (input && target) {
    target->set_data_readers_from_layer(input);
  }
  return *this;
}

sequential_model::~sequential_model() {
  // Free layers
  for (size_t l = 0; l < m_layers.size(); ++l) {
    if (m_layers[l])
      delete m_layers[l];
  }
}

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
  for (size_t l = 1; l < m_layers.size(); l++)
    if (!m_layers[l]->saveToCheckpoint(fd, filename, bytes)) {
      return false;
    }

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

  for (size_t l = 1; l < m_layers.size(); l++) {
    if (! m_layers[l]->loadFromCheckpoint(fd, filename, bytes)) {
      return false;
    }
  }

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
  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->saveToCheckpointShared(p)) {
      return false;
    }
  }

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

  // read in each layer
  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->loadFromCheckpointShared(p)) {
      return false;
    }
  }

  return true;
}

int sequential_model::num_previous_neurons() {
  if (m_layers.size() == 0) {
    return -1;
  }
  Layer *prev_layer = m_layers.back();
  return prev_layer->get_num_neurons();
}

int sequential_model::add(Layer *new_layer) {
  const uint layer_index = m_layers.size();
  new_layer->set_index(layer_index);
  m_layers.push_back(new_layer);
  return layer_index;
}

void sequential_model::remove(int index) {
  if (m_layers[index])
    delete m_layers[index];
  m_layers.erase(m_layers.begin()+index);
}

void sequential_model::insert(int index, Layer *new_layer) {
  m_layers.insert(m_layers.begin()+index, new_layer);
}

Layer *sequential_model::swap(int index, Layer *new_layer) {
  Layer *tmp = m_layers[index];
  m_layers[index] = new_layer;
  return tmp;
}

void sequential_model::setup() {
  setup_subset(0, 0);
}

void sequential_model::setup_subset(int start_index, int end_index) {
  if(end_index <= 0) {
    end_index = m_layers.size();
  }

  // Setup each layer
  for (int l=start_index; l<end_index; ++l) {
    m_layers[l]->set_neural_network_model(this); /// Provide a reverse point from each layer to the model
    const Layer* prev_layer = l > 0 ? m_layers[l-1] : nullptr;
    const Layer* next_layer = l < end_index-1 ? m_layers[l+1] : nullptr;
    m_layers[l]->add_parent_layer(prev_layer);
    m_layers[l]->add_child_layer(next_layer);
    m_layers[l]->setup();
    m_layers[l]->check_setup();
    m_layers[l]->set_index(l);
    if (m_comm->am_world_master()) {
      string description = m_layers[l]->get_description();
      std::cout << std::setw(3) << l << ":[" << std::setw(18) << m_layers[l]->get_name() <<  "] Set up a layer with input " << std::setw(7) << m_layers[l]->get_num_prev_neurons() << " and " << std::setw(7) << m_layers[l]->get_num_neurons() << " neurons.";
      std::string s = m_layers[l]->get_topo_description();
      if(s != "") {
        std::cout << " (" << s << ")";
      }
      std::cout << std::endl;
    }
  }

  // Set up callbacks
  setup_callbacks();
}

bool sequential_model::at_epoch_start() {
  // use mini batch index in data reader to signify start of epoch
  io_layer *input = (io_layer *) m_layers[0];
  bool flag = input->at_new_epoch();
  return flag;
}

/// Check if the model has a valid data set for the execution mode
bool sequential_model::is_execution_mode_valid(execution_mode mode) {
  io_layer *input = (io_layer *) m_layers[0];
  bool flag = input->is_execution_mode_valid(mode);
  return flag;
}

}  // namespace lbann
