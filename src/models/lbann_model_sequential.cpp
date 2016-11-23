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
#include "lbann/layers/lbann_layer_convolutional.hpp"
#include "lbann/layers/lbann_layer_pooling.hpp"
#include "lbann/layers/lbann_layer_fully_connected.hpp"
#include "lbann/layers/lbann_layer_softmax.hpp"
#include "lbann/optimizers/lbann_optimizer.hpp"
#include "lbann/optimizers/lbann_optimizer_sgd.hpp"
#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "mpi.h"

using namespace std;
using namespace El;

lbann::sequential_model::sequential_model(const uint mini_batch_size,
                                          lbann_comm* comm,
                                          objective_fn* obj_fn,
                                          layer_factory* _layer_fac,
                                          Optimizer_factory* _optimizer_fac)
  : model(comm, obj_fn),
    m_mini_batch_size(mini_batch_size),
    layer_fac(_layer_fac),
    optimizer_fac(_optimizer_fac) {}

lbann::sequential_model::~sequential_model()
{
  // Free layers
  for (size_t l = 0; l < m_layers.size(); ++l) {
    delete m_layers[l];
  }
}

bool lbann::sequential_model::save_to_file(const string file_dir)
{
    // get our directory name
    const char* dir = file_dir.c_str();

    // get our rank and the number of ranks
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    // report how long this takes
    Timer timer;

    // start timer
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
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
        if (!m_layers[l]->saveToFile(-1, dir))
            return false;

#if 0
    // define filename for this rank
    char filename[256];
    sprintf(filename, "%s/params.%d", dir, rank);

    // open the file for writing
    mode_t mode = S_IWUSR | S_IRUSR | S_IWGRP | S_IRGRP;
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, mode);
    int open_success = (fd != -1);
    if (! open_success) {
        fprintf(stderr, "ERROR: Failed to create file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }

    // determine whether everyone opened their file
    int all_success;
    MPI_Allreduce(&open_success, &all_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (! all_success) {
        // someone failed to create their file
        // TODO: delete our file if we created one?
        return false;
    }

    // write number of ranks (we'll check this on read)
    ssize_t write_rc = write(fd, &ranks, sizeof(int));
    if (write_rc != sizeof(int)) {
        fprintf(stderr, "ERROR: Failed to write number of ranks to file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }

    // write number of layers (we'll check this on read)
    int layers = m_layers.size();
    write_rc = write(fd, &layers, sizeof(int));
    if (write_rc != sizeof(int)) {
        fprintf(stderr, "ERROR: Failed to write number of layers to file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }

    // write out details for each layer
    for (size_t l = 1; l < m_layers.size(); l++)
        if (!m_layers[l]->saveToFile(fd, filename))
            return false;

    // fsync file
    int fsync_rc = fsync(fd);
    if (fsync_rc == -1) {
        fprintf(stderr, "ERROR: Failed to fsync file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }

    // close our file
    int close_rc = close(fd);
    if (close_rc == -1) {
        fprintf(stderr, "ERROR: Failed to close file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }
#endif

    // stop timer
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double secs = timer.Stop();
        printf("Saved parameters to %s (%f secs)\n", dir, secs);
        fflush(stdout);
    }

    return true;
}

bool lbann::sequential_model::load_from_file(const string file_dir)
{
    // get our directory name
    const char* dir = file_dir.c_str();

    // get our rank and the number of ranks
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    // report how long this takes
    Timer timer;

    // start timer
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        timer.Start();
        printf("Loading parameters from %s ...\n", dir);
        fflush(stdout);
    }

    for (size_t l = 1; l < m_layers.size(); l++)
        if (!m_layers[l]->loadFromFile(-1, dir))
            return false;

#if 0
    // get our rank and the number of ranks
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    // define filename for this rank
    char filename[256];
    sprintf(filename, "%s/params.%d", dir, rank);

    // open the file for reading
    int fd = open(filename, O_RDONLY);
    int open_success = (fd != -1);
    if (! open_success) {
        fprintf(stderr, "ERROR: Failed to open file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }

    // determine whether everyone opened their file
    int all_success;
    MPI_Allreduce(&open_success, &all_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (! all_success) {
        // someone failed to open their file
        return false;
    }

    // read number of ranks
    int file_ranks;
    ssize_t read_rc = read(fd, &file_ranks, sizeof(int));
    if (read_rc != sizeof(int)) {
        fprintf(stderr, "ERROR: Failed to read number of ranks from file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }

    if (file_ranks != ranks) {
    }

    // read number of layers
    int file_layers;
    read_rc = read(fd, &file_layers, sizeof(int));
    if (read_rc != sizeof(int)) {
        fprintf(stderr, "ERROR: Failed to read number of layers from file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }

    if (file_layers != m_layers.size()) {
    }

    for (size_t l = 1; l < m_layers.size(); l++)
        if (!m_layers[l]->loadFromFile(fd, filename))
            return false;

    // close our file
    int close_rc = close(fd);
    if (close_rc == -1) {
        fprintf(stderr, "ERROR: Failed to close file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }
#endif

    // stop timer
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double secs = timer.Stop();
        printf("Loaded parameters from %s (%f secs)\n", dir, secs);
        fflush(stdout);
    }

    return true;
}

bool lbann::sequential_model::save_to_checkpoint(int fd, const char* filename, uint64_t* bytes)
{
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
        if (!m_layers[l]->saveToCheckpoint(fd, filename, bytes))
            return false;

    return true;
}

bool lbann::sequential_model::load_from_checkpoint(int fd, const char* filename, uint64_t* bytes)
{
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

    for (size_t l = 1; l < m_layers.size(); l++)
        if (!m_layers[l]->loadFromCheckpoint(fd, filename, bytes))
            return false;

    return true;
}

bool lbann::sequential_model::save_to_checkpoint_shared(const char* dir, uint64_t* bytes)
{
    // write a single header describing layers and sizes?

    // get our rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // have rank 0 write the network file
    if (rank == 0) {
        // define filename for training state
        char filename[1024];
        sprintf(filename, "%s/network", dir);

        // open the file for writing
        int fd = lbann::openwrite(filename);

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

        // close our file
        lbann::closewrite(fd, filename);
    }

    // write out details for each layer
    for (size_t l = 1; l < m_layers.size(); l++)
        if (!m_layers[l]->saveToCheckpointShared(dir, bytes))
            return false;

    return true;
}

bool lbann::sequential_model::load_from_checkpoint_shared(const char* dir, uint64_t* bytes)
{
    // get our rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // have rank 0 read the network file
    int file_layers = -1;
    if (rank == 0) {
        // define filename for training state
        char filename[1024];
        sprintf(filename, "%s/network", dir);

        // open the file for writing
        int fd = lbann::openread(filename);
        if (fd != -1) {
            // read number of layers
            int read_rc = read(fd, &file_layers, sizeof(int));
            if (read_rc != sizeof(int)) {
                fprintf(stderr, "ERROR: Failed to read number of layers from file `%s' (%d: %s) @ %s:%d\n",
                        filename, errno, strerror(errno), __FILE__, __LINE__
                );
                fflush(stderr);
            }
            *bytes += read_rc;

            // close our file
            lbann::closeread(fd, filename);
        }
    }
    MPI_Bcast(&file_layers, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (file_layers != m_layers.size()) {
        // error!
    }

    for (size_t l = 1; l < m_layers.size(); l++)
        if (!m_layers[l]->loadFromCheckpointShared(dir, bytes))
            return false;

    return true;
}

uint lbann::sequential_model::add(const std::string layer_name,
                                  const int layer_dim,
                                  const activation_type activation,
                                  const weight_initialization init,
                                  std::vector<regularizer*> regularizers)
{
    const int layer_index = m_layers.size();
    Optimizer *optimizer = optimizer_fac->create_optimizer();

    // Get properties of previous layer
    int prev_layer_dim = -1;
    int prev_layer_index = -1;
    if(m_layers.size() != 0) {
      Layer* prev_layer = m_layers.back();
      prev_layer_dim = prev_layer->NumNeurons;
      prev_layer_index = prev_layer->Index;
    }

    if (comm->am_model_master()) {
      std::cout << "Adding a layer with input " << prev_layer_dim
                << " and index " << layer_index
                << " prev layer index " << prev_layer_index << std::endl;
    }

    if(layer_name.compare("FullyConnected") == 0) {
      Layer* new_layer
        = layer_fac->create_layer<FullyConnectedLayer>("FullyConnected",
                                                       layer_index,
                                                       prev_layer_dim,
                                                       layer_dim,
                                                       m_mini_batch_size,
                                                       activation, init,
                                                       comm,
                                                       optimizer,
                                                       regularizers);
      m_layers.push_back(new_layer);
    } else if(layer_name.compare("Softmax") == 0) {
      Layer* new_layer
        = layer_fac->create_layer<SoftmaxLayer>("Softmax",
                                                layer_index,
                                                prev_layer_dim,
                                                layer_dim,
                                                m_mini_batch_size,
                                                init,
                                                comm,
                                                optimizer);
      m_layers.push_back(new_layer);
    } else {
      std::cout << "Unknown layer type " << layer_name << std::endl;
    }

    return layer_index;
}

uint lbann::sequential_model::add(Layer *new_layer)
{
  const uint layer_index = m_layers.size();
  new_layer->Index = layer_index;
  m_layers.push_back(new_layer);
  new_layer->Index = layer_index;
  return layer_index;
}

void lbann::sequential_model::remove(int index)
{
  delete m_layers[index];
  m_layers.erase(m_layers.begin()+index);
}

void lbann::sequential_model::insert(int index, Layer *new_layer)
{
  m_layers.insert(m_layers.begin()+index, new_layer);
}

lbann::Layer* lbann::sequential_model::swap(int index, Layer *new_layer) {
  Layer* tmp = m_layers[index];
  m_layers[index] = new_layer;
  return tmp;
}

/*void lbann::sequential_model::setup()
{
  // Setup each layer
  int prev_layer_dim = -1;
  for (size_t l = 0; l < m_layers.size(); ++l) {
    if (comm->am_model_master()) {
      cout << "Setting up a layer with input " << prev_layer_dim << " and index " << m_layers[l]->Index << endl;
    }
    m_layers[l]->neural_network_model = this; /// Provide a reverse point from each layer to the model
    m_layers[l]->setup(prev_layer_dim);
    prev_layer_dim = m_layers[l]->NumNeurons;
  }

  // Establish the forward pass input pointers
  // Note: the first layer doesn't require input
  for (size_t l = 1; l < m_layers.size(); ++l) {
    m_layers[l]->setup_fp_input(m_layers[l-1]->fp_output());
  }

  // Establish the backward pass input pointers
  // Note: the last layer doesn't require input
  for (int l = m_layers.size()-2; l >= 0; --l) {
    m_layers[l]->setup_bp_input(m_layers[l+1]->bp_output());
  }

  // Set up callbacks
  setup_callbacks();
}*/

void lbann::sequential_model::setup(size_t start_index)
{
  // Setup each layer
  int prev_layer_dim, fp_index,bp_index;
  if(start_index > 0) {
    prev_layer_dim = m_layers[start_index-1]->NumNeurons;
    fp_index = start_index;
    bp_index = start_index-1;
  }else {
    prev_layer_dim = -1;
    fp_index = 1;
    bp_index = 0;
  }

  for (size_t l = start_index; l < m_layers.size(); ++l) {
    if (comm->am_model_master()) {
      cout << "Setting up a layer with input " << prev_layer_dim << " and index " << l << endl;
    }
    m_layers[l]->neural_network_model = this; /// Provide a reverse point from each layer to the model
    m_layers[l]->setup(prev_layer_dim);
    prev_layer_dim = m_layers[l]->NumNeurons;
    m_layers[l]->Index = l;
  }

  // Establish the forward pass input pointers
  // Note: the first layer doesn't require input
  for (size_t l = fp_index; l < m_layers.size(); ++l) {
    if (comm->am_model_master()) {
      //cout << "Setting up a fp input of layer " << l << "as layer " << l-1 << endl;
    }
    m_layers[l]->setup_fp_input(m_layers[l-1]->fp_output());
  }

  // Establish the backward pass input pointers
  // Note: the last layer doens't require input
  for (int l = m_layers.size()-2; l >= bp_index; --l) {
    if (comm->am_model_master()) {
      //cout << "Setting up a bp input of layer " << l << "as layer " << l+1 << endl;
    }
    m_layers[l]->setup_bp_input(m_layers[l+1]->bp_output());
  }

  // Set up callbacks
  setup_callbacks();
}



#if 0
DistMat* lbann::sequential_model::predict_mini_batch(DistMat* X)
{
    // setup input for forward, backward pass (last/additional row should always be 1)
  //    this->setup(X, NULL);

    // forward propagation (mini-batch)
    DataType L2NormSum = 0;
    for (size_t l = 1; l < m_layers.size(); l++) {
        L2NormSum = m_layers[l]->forwardProp(L2NormSum);
    }

    return m_layers[m_layers.size()-1]->fp_output();
}
#endif
