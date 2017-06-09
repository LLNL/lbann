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

#include "lbann/models/lbann_model_sequential.hpp"
#include "lbann/layers/lbann_io_layer.hpp"
#include "lbann/layers/lbann_layer_convolutional.hpp"
#include "lbann/layers/lbann_layer_pooling.hpp"
#include "lbann/layers/lbann_layer_fully_connected.hpp"
#include "lbann/layers/lbann_layer_softmax.hpp"
#include "lbann/optimizers/lbann_optimizer.hpp"
#include "lbann/optimizers/lbann_optimizer_sgd.hpp"
#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"
#include "lbann/io/lbann_persist.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "mpi.h"

using namespace std;
using namespace El;

lbann::sequential_model::sequential_model(const uint mini_batch_size,
    lbann_comm *comm,
    objective_functions::objective_fn *obj_fn,
    optimizer_factory *optimizer_fac)
  : model(comm, obj_fn, optimizer_fac),
    m_mini_batch_size(mini_batch_size)
    {}

lbann::sequential_model::~sequential_model() {
  // Free layers
  for (size_t l = 0; l < m_layers.size(); ++l) {
    delete m_layers[l];
  }
}

bool lbann::sequential_model::save_to_file(const string file_dir) {
  // get our directory name
  const char *dir = file_dir.c_str();

  // report how long this takes
  Timer timer;

  // start timer
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_rank == 0) {
    timer.Start();
    printf("Saving parameters to %s ...\n", dir);
    fflush(stdout);
  }

  // create directory to hold files
  int mkdir_success = lbann::makedir(dir);
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
  if (m_rank == 0) {
    double secs = timer.Stop();
    printf("Saved parameters to %s (%f secs)\n", dir, secs);
    fflush(stdout);
  }

  return true;
}

bool lbann::sequential_model::load_from_file(const string file_dir) {
  // get our directory name
  const char *dir = file_dir.c_str();

  // report how long this takes
  Timer timer;

  // start timer
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_rank == 0) {
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
  if (m_rank == 0) {
    double secs = timer.Stop();
    printf("Loaded parameters from %s (%f secs)\n", dir, secs);
    fflush(stdout);
  }

  return true;
}

bool lbann::sequential_model::save_to_checkpoint(int fd, const char *filename, uint64_t *bytes) {
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

bool lbann::sequential_model::load_from_checkpoint(int fd, const char *filename, uint64_t *bytes) {
  // read number of layers
  int file_layers;
  int read_rc = read(fd, &file_layers, sizeof(int));
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

bool lbann::sequential_model::save_to_checkpoint_shared(lbann::persist& p) {
  // write parameters from base class first
  model::save_to_checkpoint_shared(p);

  // write a single header describing layers and sizes?

  // have rank 0 write the network file
  if (p.m_rank == 0) {
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

bool lbann::sequential_model::load_from_checkpoint_shared(lbann::persist& p) {
  // read parameters from base class first
  model::load_from_checkpoint_shared(p);

  // have rank 0 read the network file
  struct lbann_model_sequential_header header;
  if (p.m_rank == 0) {
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

int lbann::sequential_model::num_previous_neurons() {
  if (m_layers.size() == 0) {
    return -1;
  }
  Layer *prev_layer = m_layers.back();
  return prev_layer->NumNeurons;
}

uint lbann::sequential_model::add(Layer *new_layer) {
  const uint layer_index = m_layers.size();
  new_layer->Index = layer_index;
  m_layers.push_back(new_layer);
  new_layer->Index = layer_index;
  return layer_index;
}

void lbann::sequential_model::remove(int index) {
  delete m_layers[index];
  m_layers.erase(m_layers.begin()+index);
}

void lbann::sequential_model::insert(int index, Layer *new_layer) {
  m_layers.insert(m_layers.begin()+index, new_layer);
}

lbann::Layer *lbann::sequential_model::swap(int index, Layer *new_layer) {
  Layer *tmp = m_layers[index];
  m_layers[index] = new_layer;
  return tmp;
}

void lbann::sequential_model::set_fp_input(size_t start_index, size_t end_index) {
  // Get properties from previous layers
  // Note: the first layer has no previous layer
  for (Int l=Max(start_index,1); l<end_index; ++l) {
    m_layers[l]->set_prev_layer_type(m_layers[l-1]->m_type);
    m_layers[l]->setup_fp_input(m_layers[l-1]->fp_output());
#ifdef __LIB_CUDNN
    m_layers[l]->setup_fp_input_d(m_layers[l-1]->fp_output_d());
    m_layers[l]->set_prev_layer_using_gpus(m_layers[l-1]->using_gpus());
#endif
  }
}

void lbann::sequential_model::set_bp_input(size_t start_index, size_t end_index) {
  // Get properties from next layers
  // Note: the last layer has no next layer
  for (Int l=end_index-2; l>=Max(start_index-1,0); --l) {
    m_layers[l]->set_next_layer_type(m_layers[l+1]->m_type);
    m_layers[l]->setup_bp_input(m_layers[l+1]->bp_output());
#ifdef __LIB_CUDNN
    m_layers[l]->setup_bp_input_d(m_layers[l+1]->bp_output_d());
    m_layers[l]->set_next_layer_using_gpus(m_layers[l+1]->using_gpus());
#endif
  }
}

void lbann::sequential_model::setup(size_t start_index,size_t end_index) {
  if(end_index <= 0) {
    end_index = m_layers.size();
  }

  // Get properties from adjacent layers
  set_fp_input(start_index, end_index);
  set_bp_input(start_index, end_index);

  // Setup each layer
  int prev_layer_dim = start_index > 0 ? m_layers[start_index-1]->NumNeurons : -1;
  for (Int l=start_index; l<end_index; ++l) {
    if (comm->am_model_master()) {
      cout << l << ":[" << _layer_type_to_string(m_layers[l]->m_type) <<  "] Setting up a layer with input " << prev_layer_dim << " and " << m_layers[l]->NumNeurons << " neurons."  << endl;
    }
    m_layers[l]->neural_network_model = this; /// Provide a reverse point from each layer to the model
    m_layers[l]->setup(prev_layer_dim);
    m_layers[l]->check_setup();
    prev_layer_dim = m_layers[l]->NumNeurons;
    m_layers[l]->Index = l;
  }

  // Set up callbacks
  setup_callbacks();
}

bool lbann::sequential_model::at_epoch_start() {
  // use mini batch index in data reader to signify start of epoch
  lbann::io_layer *input = (lbann::io_layer *) m_layers[0];
  bool flag = input->at_new_epoch();
  return flag;
}

