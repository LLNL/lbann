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

lbann::Sequential::Sequential(Optimizer_factory *optimizer_factory, const uint MiniBatchSize, lbann_comm* comm,layer_factory* layer_fac)
  : Model(comm), optimizer_factory(optimizer_factory), MiniBatchSize(MiniBatchSize), lfac(layer_fac)
{
}

lbann::Sequential::~Sequential()
{
    // free neural network (layers)
    for (size_t l = 0; l < Layers.size(); l++) {
        delete Layers[l];
    }
}

bool lbann::Sequential::saveToFile(string FileDir)
{
    // get our directory name
    const char* dir = FileDir.c_str();

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
    for (size_t l = 1; l < Layers.size(); l++)
        if (!Layers[l]->saveToFile(-1, dir))
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
    int layers = Layers.size();
    write_rc = write(fd, &layers, sizeof(int));
    if (write_rc != sizeof(int)) {
        fprintf(stderr, "ERROR: Failed to write number of layers to file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }

    // write out details for each layer
    for (size_t l = 1; l < Layers.size(); l++)
        if (!Layers[l]->saveToFile(fd, filename))
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

bool lbann::Sequential::loadFromFile(string FileDir)
{
    // get our directory name
    const char* dir = FileDir.c_str();

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

    for (size_t l = 1; l < Layers.size(); l++)
        if (!Layers[l]->loadFromFile(-1, dir))
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

    if (file_layers != Layers.size()) {
    }

    for (size_t l = 1; l < Layers.size(); l++)
        if (!Layers[l]->loadFromFile(fd, filename))
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

bool lbann::Sequential::saveToCheckpoint(int fd, const char* filename, uint64_t* bytes)
{
    // write number of layers (we'll check this on read)
    int layers = Layers.size();
    int write_rc = write(fd, &layers, sizeof(int));
    if (write_rc != sizeof(int)) {
        fprintf(stderr, "ERROR: Failed to write number of layers to file `%s' (%d: %s) @ %s:%d\n",
                filename, errno, strerror(errno), __FILE__, __LINE__
        );
        fflush(stderr);
    }
    *bytes += write_rc;

    // write out details for each layer
    for (size_t l = 1; l < Layers.size(); l++)
        if (!Layers[l]->saveToCheckpoint(fd, filename, bytes))
            return false;

    return true;
}

bool lbann::Sequential::loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes)
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

    if (file_layers != Layers.size()) {
        // error!
    }

    for (size_t l = 1; l < Layers.size(); l++)
        if (!Layers[l]->loadFromCheckpoint(fd, filename, bytes))
            return false;

    return true;
}

bool lbann::Sequential::saveToCheckpointShared(const char* dir, uint64_t* bytes)
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
        int layers = Layers.size();
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
    for (size_t l = 1; l < Layers.size(); l++)
        if (!Layers[l]->saveToCheckpointShared(dir, bytes))
            return false;

    return true;
}

bool lbann::Sequential::loadFromCheckpointShared(const char* dir, uint64_t* bytes)
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

    if (file_layers != Layers.size()) {
        // error!
    }

    for (size_t l = 1; l < Layers.size(); l++)
        if (!Layers[l]->loadFromCheckpointShared(dir, bytes))
            return false;

    return true;
}

uint lbann::Sequential::add(std::string layerName,
                            int LayerDim,
                            activation_type ActivationType,
                            weight_initialization init,
                            std::vector<regularizer*> regs)
{
    int prevLayerDim = -1;
    int layerIndex = Layers.size();
    int prevLayerIndex = -1;
    Optimizer *optimizer = optimizer_factory->create_optimizer();

    if(Layers.size() != 0) {
      Layer *prev = Layers.back();
      prevLayerDim = prev->NumNeurons;
      prevLayerIndex = prev->Index;
    }

    if (comm->am_model_master()) {
      cout << "Adding a layer with input " << prevLayerDim << " and index " << layerIndex << " prev layer index " << prevLayerIndex << endl;
    }

    if(layerName.compare("FullyConnected") == 0) {
      // initalize neural network (layers)
      // TODO: user-selected weight initialization
      Layers.push_back(lfac->create_layer<FullyConnectedLayer>("FullyConnected", layerIndex, prevLayerDim, LayerDim, MiniBatchSize, ActivationType, init, comm, optimizer, regs));
    }else if(layerName.compare("SoftMax") == 0) {
      Layers.push_back(lfac->create_layer<SoftmaxLayer>("SoftMax",layerIndex, prevLayerDim, LayerDim, MiniBatchSize, init, comm, optimizer));
    }else {
      std::cout << "Unknown layer type " << layerName << std::endl;
    }

    return layerIndex;
}

uint lbann::Sequential::add(Layer *new_layer)
{
  uint layer_index = Layers.size();
  Layers.push_back(new_layer);
  return layer_index;
}

lbann::Layer* lbann::Sequential::remove(int index)
{
  Layer *tmp = Layers[index];
  Layers.erase(Layers.begin()+index);
  return tmp;
}

void lbann::Sequential::insert(int index, Layer *new_layer)
{
  it = Layers.begin();
  it = Layers.insert(it+index, new_layer);
  return;
}

lbann::Layer* lbann::Sequential::swap(int index, Layer *new_layer) {
  Layer *tmp = Layers[index];
  Layers.at(index) = new_layer;
  // Layer *tmp = remove(index);
  // insert(index, new_layer);
  return tmp;
}

#if 0
void lbann::Sequential::add(vector<Layer *> new_layers)
{
  for_each in new_layers {
    Layers.push_back(new_layer);
  }
    return;
}
#endif

/**
 * Initialize the all of the model's layers.  This includes:
 *   - setting up the input dimensions for each layer
 *   - allocating and initializing memory
 *   - passing pointers for input data structures for both forward and backwards propagation passes
 */
void lbann::Sequential::setup()
{
  // Setup each layer
  int prevLayerDim = -1;
  for (size_t l = 0; l < Layers.size(); l++) {
    if (comm->am_model_master()) {
      cout << "Setting up a layer with input " << prevLayerDim << " and index " << l << endl;
    }
    Layers[l]->setup(prevLayerDim);
    prevLayerDim = Layers[l]->NumNeurons;
  }

  /// Establish the forward pass input pointers
  /// The 0'th layer cannot require any input
  for (size_t l = 1; l < Layers.size(); l++) {
    Layers[l]->setup_fp_input(Layers[l-1]->fp_output());
  }

  /// Establish the backwards pass input pointers
  /// The n'th layer cannot require any input
  for (int l = Layers.size()-2; l >= 0; l--) {
    Layers[l]->setup_bp_input(Layers[l+1]->bp_output());
  }

  // Set up callbacks.
  setup_callbacks();
}

#if 0
DistMat* lbann::Sequential::predictBatch(DistMat* X)
{
    // setup input for forward, backward pass (last/additional row should always be 1)
  //    this->setup(X, NULL);

    // forward propagation (mini-batch)
    DataType L2NormSum = 0;
    for (size_t l = 1; l < Layers.size(); l++) {
        L2NormSum = Layers[l]->forwardProp(L2NormSum);
    }

    return Layers[Layers.size()-1]->fp_output();
}
#endif
