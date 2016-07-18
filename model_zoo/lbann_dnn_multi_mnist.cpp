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
// lbann_dnn_multi_mnist.cpp - DNN application for mnist with multiple, parallel models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/data_readers/lbann_data_reader_mnist.hpp"
#include "lbann/callbacks/lbann_callback_imcomm.hpp"

using namespace std;
using namespace lbann;
using namespace El;

// layer definition
const std::vector<int> g_LayerDim = {784, 100, 30, 10};
const uint g_NumLayers = g_LayerDim.size(); // # layers

int main(int argc, char* argv[])
{
  // El initialization (similar to MPI_Init)
  Initialize(argc, argv);

  try {
    // Get data files.
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

    // Set up parameter defaults.
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/MNIST/";
    trainParams.EpochCount = 20;
    trainParams.MBSize = 10;
    trainParams.LearnRate = 0.0001;
    trainParams.DropOut = -1.0f;
    trainParams.ProcsPerModel = 12;  // Use one Catalyst node.
    trainParams.IntermodelCommMethod = static_cast<int>(
      lbann_callback_imcomm::COMPRESSED_ADAPTIVE_THRESH_QUANTIZATION);
    PerformanceParams perfParams;
    perfParams.BlockSize = 256;

    // Parse command-line inputs
    trainParams.parse_params();
    perfParams.parse_params();

    ProcessInput();
    PrintInputReport();

    // set algorithmic blocksize
    SetBlocksize(perfParams.BlockSize);

    // Set up the communicator and get the grid.
    lbann_comm* comm = new lbann_comm(trainParams.ProcsPerModel);
    Grid& grid = comm->get_model_grid();
    if (comm->am_world_master()) {
      cout << "Number of models: " << comm->get_num_models() << 
        " (" << comm->get_procs_per_model() << " procs per model)" << endl;
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

    ///////////////////////////////////////////////////////////////////
    // load training data (MNIST)
    ///////////////////////////////////////////////////////////////////
    DataReader_MNIST mnist_trainset(trainParams.MBSize);
    if (!mnist_trainset.load(trainParams.DatasetRootDir,
                             g_MNIST_TrainImageFile,
                             g_MNIST_TrainLabelFile)) {
      if (comm->am_world_master()) {
        cout << "MNIST train data error" << endl;
      }
      return -1;
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data (MNIST)
    ///////////////////////////////////////////////////////////////////
    DataReader_MNIST mnist_testset(trainParams.MBSize);
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
    Optimizer_factory *optimizer;
    if (trainParams.LearnRateMethod == 1) { // Adagrad
      optimizer = new Adagrad_factory(comm, trainParams.LearnRate);
    } else if (trainParams.LearnRateMethod == 2) { // RMSprop
      optimizer = new RMSprop_factory(comm/*, trainParams.LearnRate*/);
    } else {
      optimizer = new SGD_factory(comm, trainParams.LearnRate, 0.9,
                                  trainParams.LrDecayRate, true);
    }

    layer_factory* lfac = new layer_factory();
    Dnn dnn(trainParams.MBSize, trainParams.Lambda, optimizer, comm, lfac);
    input_layer *input_layer = new input_layer_distributed_minibatch(
      comm, (int) trainParams.MBSize, &mnist_trainset, &mnist_testset);
    //input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, &mnist_trainset, &mnist_testset);
    dnn.add(input_layer);
    uint fcidx1 = dnn.add(
      "FullyConnected", 100, trainParams.ActivationType,
      {new dropout(trainParams.DropOut)});
    uint fcidx2 = dnn.add(
      "FullyConnected", 30, trainParams.ActivationType,
      {new dropout(trainParams.DropOut)});
    uint smidx = dnn.add("SoftMax", 10);
    target_layer *target_layer = new target_layer_distributed_minibatch(
      comm, (int) trainParams.MBSize, &mnist_trainset, &mnist_testset, true);
    //target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, &mnist_trainset, &mnist_testset, true);
    dnn.add(target_layer);

    lbann_summary summarizer(trainParams.SummaryDir, comm);
    // Print out information for each epoch.
    lbann_callback_print print_cb;
    dnn.add_callback(&print_cb);
    // Record training time information.
    lbann_callback_timer timer_cb(&summarizer);
    dnn.add_callback(&timer_cb);
    // Summarize information to Tensorboard.
    lbann_callback_summary summary_cb(&summarizer, 25);
    dnn.add_callback(&summary_cb);
    // Do global inter-model updates.
    lbann_callback_imcomm imcomm_cb(
      static_cast<lbann_callback_imcomm::comm_type>(
        trainParams.IntermodelCommMethod),
      {fcidx1, fcidx2, smidx}, &summarizer);
    dnn.add_callback(&imcomm_cb);

    if (comm->am_world_master()) {
      cout << "Layer initialized:" << endl;
      for (uint n = 0; n < g_NumLayers; n++) {
        cout << "\tLayer[" << n << "]: " << g_LayerDim[n] << endl;
      }
      cout << endl;
    }

    if (comm->am_world_master()) {
      cout << "Parameter settings:" << endl;
      cout << "\tMini-batch size: " << trainParams.MBSize << endl;
      cout << "\tLearning rate: " << trainParams.LearnRate << endl << endl;
      cout << "\tEpoch count: " << trainParams.EpochCount << endl;
    }

    comm->global_barrier();

    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////

    // Initialize the model's data structures
    dnn.setup();

    comm->global_barrier();

    // train/test
    for (int t = 0; t < trainParams.EpochCount; t++) {
      dnn.train(1);
      DataType accuracy = dnn.evaluate();
    }
  }
  catch (exception& e) { ReportException(e); }

  // free all resources by El and MPI
  Finalize();

  return 0;
}
