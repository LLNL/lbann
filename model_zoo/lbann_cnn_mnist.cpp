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
// lbann_dnn_mnist.cpp - DNN application for mnist
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_mnist.hpp"
#include "lbann/lbann.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

using namespace std;
using namespace lbann;
using namespace El;

// layer definition
const std::vector<int> g_LayerDim = {784, 100, 30, 10};
const uint g_NumLayers = g_LayerDim.size(); // # layers

/// Main function
int main(int argc, char* argv[])
{
    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);

    try {

        // Get data files
        const string g_MNIST_TrainLabelFile = Input("--train-label-file",
                                                    "MNIST training set label file",
                                                    "train-labels-idx1-ubyte");
        const string g_MNIST_TrainImageFile = Input("--train-image-file",
                                                    "MNIST training set image file",
                                                    "train-images-idx3-ubyte");
        const string g_MNIST_TestLabelFile = Input("--test-label-file",
                                                   "MNIST test set label file",
                                                   "t10k-labels-idx1-ubyte");
        const string g_MNIST_TestImageFile = Input("--test-image-file",
                                                   "MNIST test set image file",
                                                   "t10k-images-idx3-ubyte");

        // Initialize parameter defaults
        TrainingParams trainParams;
        trainParams.DatasetRootDir = "/p/lscratche/brainusr/datasets/mnist-bin";
        trainParams.EpochCount = 20;
        trainParams.MBSize = 256;
        trainParams.LearnRateMethod = 2;
        trainParams.LearnRate = 0.005;
        trainParams.ActivationType = activation_type::RELU;
        trainParams.DropOut = 0.5;
        PerformanceParams perfParams;
        perfParams.BlockSize = 256;

        // Parse command-line inputs
        trainParams.parse_params();
        perfParams.parse_params();
        ProcessInput();
        PrintInputReport();

        // Set algorithmic blocksize
        SetBlocksize(perfParams.BlockSize);

        // Set up the communicator and get the grid.
        lbann_comm* comm = new lbann_comm();
        Grid& grid = comm->get_model_grid();
        if (comm->am_world_master()) {
          cout << "Number of models: " << comm->get_num_models() << endl;
          cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
          cout << endl;
        }

        int parallel_io = perfParams.MaxParIOSize;
        if (parallel_io == 0) {
          if (comm->am_world_master()) {
            cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() <<
              " (Limited to # Processes)" << endl;
          }
          parallel_io = comm->get_procs_per_model();
        } else {
          if (comm->am_world_master()) {
            cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
          }
        }

        // load training data (MNIST)
        DataReader_MNIST mnist_trainset(trainParams.MBSize, true);
        if (!mnist_trainset.load(trainParams.DatasetRootDir,
                                 g_MNIST_TrainImageFile,
                                 g_MNIST_TrainLabelFile)) {
          if (comm->am_world_master()) {
            cout << "MNIST train data error" << endl;
          }
          return -1;
        }

        // load testing data (MNIST)
        DataReader_MNIST mnist_testset(trainParams.MBSize, true);
        if (!mnist_testset.load(trainParams.DatasetRootDir,
                                g_MNIST_TestImageFile,
                                g_MNIST_TestLabelFile)) {
          if (comm->am_world_master()) {
            cout << "MNIST Test data error" << endl;
          }
          return -1;
        }

        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////

        // Initialize optimizer
        Optimizer_factory *optimizer;
        if (trainParams.LearnRateMethod == 1) { // Adagrad
          optimizer = new Adagrad_factory(comm, trainParams.LearnRate);
        }else if (trainParams.LearnRateMethod == 2) { // RMSprop
          optimizer = new RMSprop_factory(comm/*, trainParams.LearnRate*/);
        }else {
          optimizer = new SGD_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
        }

        // Initialize network
        layer_factory* lfac = new layer_factory();
        cudnn::cudnn_manager* cudnn = new cudnn::cudnn_manager();
        Dnn dnn(trainParams.MBSize,
                trainParams.Lambda,
                optimizer, comm, lfac);
        input_layer *input_layer = new input_layer_distributed_minibatch(comm,  (int) trainParams.MBSize, &mnist_trainset, &mnist_testset);
        // input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, &mnist_trainset, &mnist_testset);
        dnn.add(input_layer);

        // First convolution layer
        {
          Optimizer* convolution_layer_optimizer = optimizer->create_optimizer(matrix_format::STAR_STAR);
          int numDims = 2;
          int inputChannels = 1;
          int inputDims[] = {28, 28};
          int outputChannels = 32;
          int filterDims[] = {3, 3};
          int convPads[] = {0, 0};
          int convStrides[] = {1, 1};
          convolutional_layer* layer
            = new convolutional_layer(1, numDims, inputChannels, inputDims,
                                      outputChannels, filterDims,
                                      convPads, convStrides,
                                      trainParams.MBSize,
                                      activation_type::RELU, 
                                      comm, convolution_layer_optimizer, 
                                      {}, cudnn);
          dnn.add(layer);
        }

        // Second convolution layer
        {
          Optimizer* convolution_layer_optimizer = optimizer->create_optimizer(matrix_format::STAR_STAR);
          int numDims = 2;
          int inputChannels = 32;
          int inputDims[] = {26, 26};
          int outputChannels = 32;
          int filterDims[] = {3, 3};
          int convPads[] = {0, 0};
          int convStrides[] = {1, 1};
          convolutional_layer* layer
            = new convolutional_layer(2, numDims, inputChannels, inputDims,
                                      outputChannels, filterDims,
                                      convPads, convStrides,
                                      trainParams.MBSize,
                                      activation_type::RELU,
                                      comm, convolution_layer_optimizer,
                                      {},
                                      cudnn);
          dnn.add(layer);
        }

        // Pooling layer
        {
          int numDims = 2;
          int channels = 32;
          int inputDim[] = {24, 24};
          int poolWindowDims[] = {2, 2};
          int poolPads[] = {0, 0};
          int poolStrides[] = {2, 2};
          int poolMode = 0;
          pooling_layer* layer
            = new pooling_layer(3, numDims, channels, inputDim,
                                poolWindowDims, poolPads, poolStrides, poolMode,
                                trainParams.MBSize, activation_type::ID,
                                comm,
                                {}, //{new dropout(trainParams.DropOut)},
                                cudnn);
          dnn.add(layer);
        }

        // This is replaced by the input layer
        dnn.add("FullyConnected", 128, trainParams.ActivationType, {new dropout(trainParams.DropOut)});
        dnn.add("SoftMax", 10);

        target_layer *target_layer = new target_layer_distributed_minibatch(comm, (int) trainParams.MBSize, &mnist_trainset, &mnist_testset, true);
        // target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, &mnist_trainset, &mnist_testset, true);
        dnn.add(target_layer);

        lbann_callback_print print_cb;
        dnn.add_callback(&print_cb);
        // lbann_callback_io io_cb({0,3});
        // dnn.add_callback(&io_cb);

        // setup network/layers
        dnn.setup();
        if (comm->am_world_master()) {
          cout << "Layer initialized" << endl;
          for (int l = 0; l < (int)dnn.Layers.size(); l++) {
            cout << "[" << l << "] " << dnn.Layers[l]->NumNeurons << endl;
          }
        }

        if (comm->am_world_master()) {
          cout << "Parameter settings:" << endl;
          cout << "\tMini-batch size: " << trainParams.MBSize << endl;
          cout << "\tLearning rate: " << trainParams.LearnRate << endl << endl;
          cout << "\tEpoch count: " << trainParams.EpochCount << endl;
        }


        // start train
        dnn.train(trainParams.EpochCount, true);

        // Free dynamically allocated memory
        // delete target_layer;  // Causes segfault
        // delete input_layer;  // Causes segfault
        // delete lfac;  // Causes segfault
        delete optimizer;
        delete comm;
        delete cudnn;

    }
    catch (exception& e) { ReportException(e); }

    // free all resources by El and MPI
    Finalize();

    return 0;
}
