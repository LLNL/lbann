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
  init_random(42);  // Deterministic initialization across every model.
  init_data_seq_random(42);
  lbann_comm* comm = NULL;

  try {
    // Get data files.
    const string g_MNIST_TrainLabelFile = Input("--train-label-file",
                                                "MNIST training set label file",
                                                std::string("train-labels-idx1-ubyte"));
    const string g_MNIST_TrainImageFile = Input("--train-image-file",
                                                "MNIST training set image file",
                                                std::string("train-images-idx3-ubyte"));
    const string g_MNIST_TestLabelFile = Input("--test-label-file",
                                               "MNIST test set label file",
                                               std::string("t10k-labels-idx1-ubyte"));
    const string g_MNIST_TestImageFile = Input("--test-image-file",
                                               "MNIST test set image file",
                                               std::string("t10k-images-idx3-ubyte"));

    // Set up parameter defaults.
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/MNIST/";
    trainParams.EpochCount = 20;
    trainParams.MBSize = 10;
    trainParams.LearnRate = 0.0001;
    trainParams.DropOut = -1.0f;
    trainParams.ProcsPerModel = 12;  // Use one Catalyst node.
    trainParams.IntermodelCommMethod = static_cast<int>(
                                                        lbann_callback_imcomm::NORMAL/*ADAPTIVE_THRESH_QUANTIZATION*/);
    trainParams.PercentageTrainingSamples = 1.0;
    trainParams.PercentageValidationSamples = 0.1;
    PerformanceParams perfParams;
    perfParams.BlockSize = 256;

    // Parse command-line inputs
    trainParams.parse_params();
    perfParams.parse_params();

    ProcessInput();
    PrintInputReport();

    // set algorithmic blocksize
    SetBlocksize(perfParams.BlockSize);

    El::GemmUseGPU(512,512,512);

    // Set up the communicator and get the grid.
    comm = new lbann_comm(trainParams.ProcsPerModel);
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
    mnist_trainset.set_file_dir(trainParams.DatasetRootDir);
    mnist_trainset.set_data_filename(g_MNIST_TrainImageFile);
    mnist_trainset.set_label_filename(g_MNIST_TrainLabelFile);
    mnist_trainset.set_validation_percent(trainParams.PercentageValidationSamples);
    mnist_trainset.load();

    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data (MNIST)
    ///////////////////////////////////////////////////////////////////
    DataReader_MNIST mnist_validation_set(mnist_trainset); // Clone the training set object
    mnist_validation_set.use_unused_index_set();

        if (comm->am_world_master()) {
          size_t num_train = mnist_trainset.getNumData();
          size_t num_validate = mnist_trainset.getNumData();
          double validate_percent = num_validate / (num_train+num_validate)*100.0;
          double train_percent = num_train / (num_train+num_validate)*100.0;
          cout << "Training using " << train_percent << "% of the training data set, which is " << mnist_trainset.getNumData() << " samples." << endl
               << "Validating training using " << validate_percent << "% of the training data set, which is " << mnist_validation_set.getNumData() << " samples." << endl;
        }

    ///////////////////////////////////////////////////////////////////
    // load testing data (MNIST)
    ///////////////////////////////////////////////////////////////////
    DataReader_MNIST mnist_testset(trainParams.MBSize);
    mnist_testset.set_file_dir(trainParams.DatasetRootDir);
    mnist_testset.set_data_filename(g_MNIST_TestImageFile);
    mnist_testset.set_label_filename(g_MNIST_TestLabelFile);
    mnist_testset.set_use_percent(trainParams.PercentageTestingSamples);
    mnist_testset.load();

    if (comm->am_world_master()) {
      cout << "Testing using " << (trainParams.PercentageTestingSamples*100) <<
        "% of the testing data set, which is " << mnist_testset.getNumData() <<
        " samples." << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////

    // Initialize optimizer
    optimizer_factory *optimizer_fac;
    if (trainParams.LearnRateMethod == 1) { // Adagrad
      optimizer_fac = new adagrad_factory(comm, trainParams.LearnRate);
    }else if (trainParams.LearnRateMethod == 2) { // RMSprop
      optimizer_fac = new rmsprop_factory(comm, trainParams.LearnRate);
    } else if (trainParams.LearnRateMethod == 3) { // Adam
      optimizer_fac = new adam_factory(comm, trainParams.LearnRate);
    } else {
      optimizer_fac = new sgd_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
    }

    layer_factory* lfac = new layer_factory();
    deep_neural_network dnn(trainParams.MBSize, comm, new objective_functions::categorical_cross_entropy(comm), lfac, optimizer_fac);
    metrics::categorical_accuracy acc(comm);
    dnn.add_metric(&acc);
    std::map<execution_mode, DataReader*> data_readers = {std::make_pair(execution_mode::training,&mnist_trainset), 
                                                          std::make_pair(execution_mode::validation, &mnist_validation_set), 
                                                          std::make_pair(execution_mode::testing, &mnist_testset)};
    //input_layer *input_layer = new input_layer_distributed_minibatch(comm, (int) trainParams.MBSize, data_readers);
    input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, (int) trainParams.MBSize, data_readers);
    dnn.add(input_layer);
    uint fcidx1 = dnn.add(
      "FullyConnected", data_layout::MODEL_PARALLEL, 1024,
      trainParams.ActivationType, weight_initialization::glorot_uniform,
      {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
    uint fcidx2 = dnn.add(
      "FullyConnected", data_layout::MODEL_PARALLEL, 1024,
      trainParams.ActivationType, weight_initialization::glorot_uniform,
      {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
    uint fcidx3 = dnn.add(
      "FullyConnected", data_layout::MODEL_PARALLEL, 1024,
      trainParams.ActivationType, weight_initialization::glorot_uniform,
      {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
    uint smidx = dnn.add(
      "Softmax", data_layout::MODEL_PARALLEL, 10,
      activation_type::ID, weight_initialization::glorot_uniform, {});
    //target_layer *target_layer = new target_layer_distributed_minibatch(comm, (int) trainParams.MBSize, data_readers, true);
    target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, (int) trainParams.MBSize, data_readers, true);
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
      {fcidx1, fcidx2, fcidx3, smidx}, &summarizer);
    dnn.add_callback(&imcomm_cb);
    lbann_callback_adaptive_learning_rate lrsched(4, 0.1f);
    dnn.add_callback(&lrsched);
    // lbann_callback_io io_cb({0,4}); // Monitor layers 0 and 4
    // dnn.add_callback(&io_cb);
    lbann_callback_debug debug_cb(execution_mode::training);
    //dnn.add_callback(&debug_cb);
    //lbann_callback_early_stopping stopping_cb(1);
    //dnn.add_callback(&stopping_cb);

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

    // Reinitialize the RNG differently for each rank.
    init_random(comm->get_rank_in_world() + 1);

    comm->global_barrier();

    // train/test
    for (int t = 0; t < trainParams.EpochCount; t++) {
      dnn.train(1, true);
      dnn.evaluate();
    }
  }
  catch (lbann_exception& e) { lbann_report_exception(e, comm); }
  catch (exception& e) { ReportException(e); }

  // free all resources by El and MPI
  Finalize();

  return 0;
}
