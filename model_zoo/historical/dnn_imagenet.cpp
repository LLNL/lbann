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
// dnn_imagenet.cpp - DNN application for image-net classification
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/layers/regularizers/dropout.hpp"
#include "lbann/data_readers/image_utils.hpp"

#include <time.h>
#include <sys/time.h>
#include <iomanip>
#include <string>

using namespace std;
using namespace lbann;
using namespace El;

//#define PARTITIONED
#if defined(PARTITIONED)
#define DATA_LAYOUT data_layout::DATA_PARALLEL
#else
#define DATA_LAYOUT data_layout::MODEL_PARALLEL
#endif

// train/test data info
const int g_SaveImageIndex[1] = {0}; // for auto encoder
//const int g_SaveImageIndex[5] = {293, 2138, 3014, 6697, 9111}; // for auto encoder
//const int g_SaveImageIndex[5] = {1000, 2000, 3000, 4000, 5000}; // for auto encoder
const string g_ImageNet_TrainDir = "resized_256x256/train/";
const string g_ImageNet_ValDir = "resized_256x256/val/";
const string g_ImageNet_TestDir = "resized_256x256/val/";
const string g_ImageNet_LabelDir = "labels/";
const int g_ImageNet_Width = 256;
const int g_ImageNet_Height = 256;

const string g_ImageNet_TrainLabelFile = "train_c0-9_01.txt";
//const string g_ImageNet_TrainLabelFile = "train_c0-9.txt";
const string g_ImageNet_ValLabelFile = "val.txt";
//const string g_ImageNet_TestLabelFile = "val_c0-9.txt"; //"test.txt";
const string g_ImageNet_TestLabelFile = "val_c0-9_01.txt"; //"test.txt";


int main(int argc, char *argv[]) {
  lbann_comm *comm = initialize(argc, argv, 42);

  try {
    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/ILSVRC2012/";
    trainParams.LearnRate = 5e-3;
    trainParams.DropOut = 0.9;
    trainParams.ProcsPerModel = 0;
    trainParams.parse_params();
    trainParams.PercentageTrainingSamples = 1.0;
    trainParams.PercentageValidationSamples = 0.2;
    PerformanceParams perfParams;
    perfParams.parse_params();
    // Read in the user specified network topology
    NetworkParams netParams;
    netParams.parse_params();
    // Get some environment variables from the launch
    SystemParams sysParams;
    sysParams.parse_params();

    // regular dense neural network or auto encoder
    //const bool g_AutoEncoder = Input("--mode", "DNN: false, AutoEncoder: true", false);

    // training settings
    int decayIterations = 1;

    bool unit_scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
    bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", true);
    bool unit_variance = Input("--unit-variance", "standardize to unit-variance", true);

    //if set to true, above three settings have no effect
    bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

    bool use_new_reader = Input("--new-reader", "use new data reader", false);

    ProcessInput();
    PrintInputReport();

    // set algorithmic blocksize
    SetBlocksize(perfParams.BlockSize);

    // create timer for performance measurement
    Timer timer_io;
    Timer timer_lbann;
    Timer timer_val;
    //double sec_all_io = 0;
    //double sec_all_lbann = 0;
    //double sec_all_val = 0;

    // Set up the communicator and get the grid.
    comm->split_models(trainParams.ProcsPerModel);
    Grid& grid = comm->get_model_grid();
    if (comm->am_world_master()) {
      cout << "Number of models: " << comm->get_num_models() << endl;
      cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
      cout << endl;
    }

    int parallel_io = perfParams.MaxParIOSize;
    //        int io_offset = 0;
    if(parallel_io == 0) {
      if(comm->am_world_master()) {
        cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() << " (Limited to # Processes)" << endl;
      }
      parallel_io = comm->get_procs_per_model();
      //          io_offset = comm->get_rank_in_model() *trainParams.MBSize;
    } else {
      if(comm->am_world_master()) {
        cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
      }
      //          parallel_io = grid.Size();
      // if(perfParams.MaxParIOSize > 1) {
      //   io_offset = comm->get_rank_in_model() *trainParams.MBSize;
      // }
    }

    parallel_io = 1;

    std::map<execution_mode, generic_data_reader *> data_readers;
    ///////////////////////////////////////////////////////////////////
    // load training data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    if (not use_new_reader) {
      if (comm->am_world_master()) {
        cout << endl << "USING imagenet_reader\n\n";
      }
      imagenet_reader *imagenet_trainset = new imagenet_reader(trainParams.MBSize, true);
      imagenet_trainset->set_firstN(false);
      imagenet_trainset->set_role("train");
      imagenet_trainset->set_master(comm->am_world_master());
      imagenet_trainset->set_rank(comm->get_rank_in_world());
      imagenet_trainset->set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TrainDir);
      imagenet_trainset->set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile);
      imagenet_trainset->set_validation_percent(trainParams.PercentageValidationSamples);
      imagenet_trainset->load();

      imagenet_trainset->scale(unit_scale);
      imagenet_trainset->subtract_mean(subtract_mean);
      imagenet_trainset->unit_variance(unit_variance);
      imagenet_trainset->z_score(z_score);

      ///////////////////////////////////////////////////////////////////
      // create a validation set from the unused training data (ImageNet)
      ///////////////////////////////////////////////////////////////////
      imagenet_reader *imagenet_validation_set = new imagenet_reader(*imagenet_trainset); // Clone the training set object
      imagenet_validation_set->set_role("validation");
      imagenet_validation_set->use_unused_index_set();

      if (comm->am_world_master()) {
        size_t num_train = imagenet_trainset->getNumData();
        size_t num_validate = imagenet_trainset->getNumData();
        double validate_percent = num_validate / (num_train+num_validate)*100.0;
        double train_percent = num_train / (num_train+num_validate)*100.0;
        cout << "Training using " << train_percent << "% of the training data set, which is " << imagenet_trainset->getNumData() << " samples." << endl
             << "Validating training using " << validate_percent << "% of the training data set, which is " << imagenet_validation_set->getNumData() << " samples." << endl;
      }

      ///////////////////////////////////////////////////////////////////
      // load testing data (ImageNet)
      ///////////////////////////////////////////////////////////////////
      imagenet_reader *imagenet_testset = new imagenet_reader(trainParams.MBSize, true);
      imagenet_testset->set_firstN(false);
      imagenet_testset->set_role("test");
      imagenet_testset->set_master(comm->am_world_master());
      imagenet_testset->set_rank(comm->get_rank_in_world());
      imagenet_testset->set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TestDir);
      imagenet_testset->set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile);
      imagenet_testset->set_use_percent(trainParams.PercentageTestingSamples);
      imagenet_testset->load();

      if (comm->am_world_master()) {
        cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << imagenet_testset->getNumData() << " samples." << endl;
      }

      imagenet_testset->scale(unit_scale);
      imagenet_testset->subtract_mean(subtract_mean);
      imagenet_testset->unit_variance(unit_variance);
      imagenet_testset->z_score(z_score);

      data_readers[execution_mode::training] = imagenet_trainset;
      data_readers[execution_mode::validation] = imagenet_validation_set;
      data_readers[execution_mode::testing] = imagenet_testset;
    } else {
      //===============================================================
      // imagenet_readerSingle
      //===============================================================
      if (comm->am_world_master()) {
        cout << endl << "USING imagenet_readerSingle\n\n";
      }
      imagenet_readerSingle *imagenet_trainset = new imagenet_readerSingle(trainParams.MBSize, true);
      imagenet_trainset->set_firstN(false);
      imagenet_trainset->set_role("train");
      imagenet_trainset->set_master(comm->am_world_master());
      imagenet_trainset->set_rank(comm->get_rank_in_world());
      imagenet_trainset->set_file_dir(trainParams.DatasetRootDir);

      stringstream ss;
      ss << "Single_" << g_ImageNet_TrainLabelFile.substr(0, g_ImageNet_TrainLabelFile.size()-4);
      imagenet_trainset->set_data_filename(ss.str());
      imagenet_trainset->set_validation_percent(trainParams.PercentageValidationSamples);

      imagenet_trainset->load();

      imagenet_trainset->scale(unit_scale);
      imagenet_trainset->subtract_mean(subtract_mean);
      imagenet_trainset->unit_variance(unit_variance);
      imagenet_trainset->z_score(z_score);

      ///////////////////////////////////////////////////////////////////
      // create a validation set from the unused training data (ImageNet)
      ///////////////////////////////////////////////////////////////////
      imagenet_readerSingle *imagenet_validation_set = new imagenet_readerSingle(*imagenet_trainset); // Clone the training set object
      imagenet_validation_set->set_role("validation");
      imagenet_validation_set->use_unused_index_set();

      if (comm->am_world_master()) {
        size_t num_train = imagenet_trainset->getNumData();
        size_t num_validate = imagenet_trainset->getNumData();
        double validate_percent = num_validate / (num_train+num_validate)*100.0;
        double train_percent = num_train / (num_train+num_validate)*100.0;
        cout << "Training using " << train_percent << "% of the training data set, which is " << imagenet_trainset->getNumData() << " samples." << endl
             << "Validating training using " << validate_percent << "% of the training data set, which is " << imagenet_validation_set->getNumData() << " samples." << endl;
      }

      ///////////////////////////////////////////////////////////////////
      // load testing data (ImageNet)
      ///////////////////////////////////////////////////////////////////
      ss.clear();
      ss.str("");
      ss << "Single_" << g_ImageNet_TestLabelFile.substr(0, g_ImageNet_TestLabelFile.size()-4);
      imagenet_readerSingle *imagenet_testset = new imagenet_readerSingle(trainParams.MBSize, true);
      imagenet_testset->set_firstN(false);
      imagenet_testset->set_role("test");
      imagenet_testset->set_master(comm->am_world_master());
      imagenet_testset->set_rank(comm->get_rank_in_world());
      imagenet_testset->set_file_dir(trainParams.DatasetRootDir);
      imagenet_testset->set_data_filename(ss.str());
      imagenet_testset->set_use_percent(trainParams.PercentageTestingSamples);
      imagenet_testset->load();

      if (comm->am_world_master()) {
        cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << imagenet_testset->getNumData() << " samples." << endl;
      }

      imagenet_testset->scale(unit_scale);
      imagenet_testset->subtract_mean(subtract_mean);
      imagenet_testset->unit_variance(unit_variance);
      imagenet_testset->z_score(z_score);

      data_readers[execution_mode::training] = imagenet_trainset;
      data_readers[execution_mode::validation] = imagenet_validation_set;
      data_readers[execution_mode::testing] = imagenet_testset;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////

    // Initialize optimizer factory
    optimizer_factory *optimizer_fac;
    if (trainParams.LearnRateMethod == 1) { // Adagrad
      optimizer_fac = new adagrad_factory(comm, trainParams.LearnRate);
    } else if (trainParams.LearnRateMethod == 2) { // RMSprop
      optimizer_fac = new rmsprop_factory(comm, trainParams.LearnRate);
    } else if (trainParams.LearnRateMethod == 3) { // Adam
      optimizer_fac = new adam_factory(comm, trainParams.LearnRate);
    } else {
      optimizer_fac = new sgd_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
    }

    deep_neural_network *dnn = NULL;
    dnn = new deep_neural_network(trainParams.MBSize, comm, new objective_functions::categorical_cross_entropy(comm), optimizer_fac);
    dnn->add_metric(new metrics::categorical_accuracy<DATA_LAYOUT>(comm));
    // input_layer *input_layer = new input_layer_distributed_minibatch(data_layout::DATA_PARALLEL, comm, trainParams.MBSize, data_readers);
#ifdef PARTITIONED
    input_layer *input_layer = new input_layer_partitioned_minibatch<>(comm, trainParams.MBSize, parallel_io, data_readers);
#else
    input_layer *input_layer = new input_layer_distributed_minibatch<DATA_LAYOUT>(comm, trainParams.MBSize, parallel_io, data_readers);
#endif
    dnn->add(input_layer);

    const int NumLayers = netParams.Network.size();
    int lcnt = 1;
    // initalize neural network (layers)
    for (int l = 0; l < NumLayers-1; l++) {
      fully_connected_layer<DATA_LAYOUT> *fc 
        = new fully_connected_layer<DATA_LAYOUT>(
          lcnt++,
          comm,
          trainParams.MBSize,
          netParams.Network[l],
          weight_initialization::glorot_uniform, 
          dnn->create_optimizer());
      dnn->add(fc);

      Layer *act = NULL;
      if (trainParams.ActivationType == 1) { // sigmoid
        act = new sigmoid_layer<DATA_LAYOUT>(lcnt++, comm, trainParams.MBSize);
      } else if (trainParams.ActivationType == 2) { // tanh
        act = new tanh_layer<DATA_LAYOUT>(lcnt++, comm, trainParams.MBSize);
      } else if (trainParams.ActivationType == 3) { // reLU
        act = new relu_layer<DATA_LAYOUT>(lcnt++, comm, trainParams.MBSize);
      } else { // ID
        act = new id_layer<DATA_LAYOUT>(lcnt++, comm, trainParams.MBSize);
      }
      dnn->add(act);

      Layer *reg = new dropout<DATA_LAYOUT>(lcnt++,
                                                comm, trainParams.MBSize,
                                                trainParams.DropOut);
      dnn->add(reg);
    }

    // softmax layer
    {
      // Fully-connected without bias before softmax.
      fully_connected_layer<DATA_LAYOUT> *fc
        = new fully_connected_layer<DATA_LAYOUT>(
          lcnt++,
          comm,
          trainParams.MBSize,
          netParams.Network[NumLayers-1],
          weight_initialization::glorot_uniform, 
          dnn->create_optimizer(),
          false);
      dnn->add(fc);
      //      get_prev_neurons_and_index( dnn, prev_num_neurons, layer_id);
      Layer *softmax 
        = new softmax_layer<DATA_LAYOUT>(
          lcnt++,
          comm,
          trainParams.MBSize,
          dnn->create_optimizer());
      dnn->add(softmax);
    }

    //target_layer *target_layer = new target_layer_distributed_minibatch(comm, trainParams.MBSize, &imagenet_trainset, &imagenet_testset, true);
#ifdef PARTITIONED
    Layer *target_layer = new target_layer_partitioned_minibatch<>(comm, trainParams.MBSize, parallel_io, data_readers, true);
#else
    Layer *target_layer = new target_layer_distributed_minibatch<DATA_LAYOUT>(comm, trainParams.MBSize, parallel_io, data_readers, true);
#endif
    dnn->add(target_layer);


    lbann_summary summarizer(trainParams.SummaryDir, comm);
    // Print out information for each epoch.
    lbann_callback_print print_cb;
    dnn->add_callback(&print_cb);
    // Record training time information.
    lbann_callback_timer timer_cb(&summarizer);
    dnn->add_callback(&timer_cb);
    // Summarize information to Tensorboard.
    lbann_callback_summary summary_cb(&summarizer, 25);
    dnn->add_callback(&summary_cb);
    // lbann_callback_io io_cb({0});
    // dnn->add_callback(&io_cb);
    lbann_callback_adaptive_learning_rate lrsched(4, 0.1f);
    dnn->add_callback(&lrsched);

    dnn->setup();

    if (grid.Rank() == 0) {
      cout << "Layer initialized:" << endl;
      for (uint n = 0; n < dnn->get_layers().size(); n++) {
        cout << "\tLayer[" << n << "]: " << dnn->get_layers()[n]->get_num_neurons() << endl;
      }
      cout << endl;

      cout << "Parameter settings:" << endl;
      cout << "\tBlock size: " << perfParams.BlockSize << endl;
      cout << "\tEpochs: " << trainParams.EpochCount << endl;
      cout << "\tMini-batch size: " << trainParams.MBSize << endl;
      cout << "\tLearning rate: " << trainParams.LearnRate << endl;
      cout << "\tEpoch count: " << trainParams.EpochCount << endl << endl;
      if(perfParams.MaxParIOSize == 0) {
        cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
      } else {
        cout << "\tMax Parallel I/O Fetch: " << perfParams.MaxParIOSize << endl;
      }
      cout << "\tDataset: " << trainParams.DatasetRootDir << endl;
    }

    // load parameters from file if available
    if (trainParams.LoadModel && trainParams.ParameterDir.length() > 0) {
      dnn->load_from_file(trainParams.ParameterDir);
    }

    mpi::Barrier(grid.Comm());


    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////

    int last_layer_size;
    last_layer_size = netParams.Network[netParams.Network.size()-1];

    //************************************************************************
    // read training state from checkpoint file if we have one
    //************************************************************************
    int epochStart = 0; // epoch number we should start at
    //int trainStart; // index into indices we should start at

    //************************************************************************
    // mainloop for train/validate
    //************************************************************************
    for (int epoch = epochStart; epoch < trainParams.EpochCount; epoch++) {
      decayIterations = 1;

      //************************************************************************
      // training epoch loop
      //************************************************************************

      dnn->train(1, true);

      dnn->evaluate(execution_mode::testing);
    }

    delete dnn;
  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);  /// Elemental exceptions
  }

  // free all resources by El and MPI
  finalize(comm);

  return 0;
}

















#if 0
int main(int argc, char *argv[]) {
  // El initialization (similar to MPI_Init)
  Initialize(argc, argv);

  try {
    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////
    TrainingParams trainParams("/p/lscratchf/brainusr/datasets/ILSVRC2012/");
    PerformanceParams perfParams;
    // Read in the user specified network topology
    NetworkParams netParams;
    // Get some environment variables from the launch
    SystemParams sysParams;

    // regular dense neural network or auto encoder
    const bool g_AutoEncoder = Input("--mode", "DNN: false, AutoEncoder: true", false);

    // int inputDimension = 65536 * 3;
    // // Add in the imagenet specific part of the topology
    // std::vector<int>::iterator it;

    // it = netParams.Network.begin();
    // netParams.Network.insert(it, inputDimension);

    // training settings
    int decayIterations = 1;

    ProcessInput();
    PrintInputReport();

    // set algorithmic blocksize
    SetBlocksize(perfParams.BlockSize);

    // create a Grid: convert MPI communicators into a 2-D process grid
    Grid grid(mpi::COMM_WORLD);
    if (grid.Rank() == 0) {
      cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
      cout << endl;
    }

    // create timer for performance measurement
    Timer timer_io;
    Timer timer_lbann;
    Timer timer_val;
    double sec_all_io = 0;
    double sec_all_lbann = 0;
    double sec_all_val = 0;

    ///////////////////////////////////////////////////////////////////
    // load training data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    imagenet_reader imagenet_trainset(trainParams.MBSize, true, grid.Rank()*trainParams.MBSize, parallel_io*trainParams.MBSize);
    if (!imagenet_trainset.load(trainParams.DatasetRootDir, g_MNIST_TrainImageFile, g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile)) {
      if (comm->am_world_master()) {
        cout << "ImageNet train data error" << endl;
      }
      return -1;
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    mnist_reader imagenet_testset(trainParams.MBSize, true, grid.Rank()*trainParams.MBSize, parallel_io*trainParams.MBSize);
    if (!imagenet_testset.load(g_MNIST_Dir, g_MNIST_TestImageFile, g_MNIST_TestLabelFile)) {
      if (comm->am_world_master()) {
        cout << "ImageNet Test data error" << endl;
      }
      return -1;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////
    optimizer_factory *optimizer_fac;
    if (trainParams.LearnRateMethod == 1) { // Adagrad
      optimizer_fac = new adagrad_factory(grid, trainParams.LearnRate);
    } else if (trainParams.LearnRateMethod == 2) { // RMSprop
      optimizer_fac = new rmsprop_factory(grid, trainParams.LearnRate);
    } else if (trainParams.LearnRateMethod == 3) { // Adam
      optimizer_fac = new adam_factory(grid, trainParams.LearnRate);
    } else {
      optimizer = new sgd_factory(grid, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
    }

    deep_neural_network *dnn = NULL;
    AutoEncoder *autoencoder = NULL;
    if (g_AutoEncoder) {
      // need to fix later!!!!!!!!!!!!!!!!!!!!!!!  netParams.Network should be separated into encoder and decoder parts
      //autoencoder = new AutoEncoder(netParams.Network, netParams.Network, false, trainParams.MBSize, trainParams.ActivationType, trainParams.DropOut, trainParams.Lambda, grid);
      autoencoder = new AutoEncoder(optimizer_fac, trainParams.MBSize, grid);
      // autoencoder.add("FullyConnected", 784, g_ActivationType, g_DropOut, trainParams.Lambda);
      // autoencoder.add("FullyConnected", 100, g_ActivationType, g_DropOut, trainParams.Lambda);
      // autoencoder.add("FullyConnected", 30, g_ActivationType, g_DropOut, trainParams.Lambda);
      // autoencoder.add("softmax", 10);
    } else {
      dnn = new deep_neural_network(optimizer_fac, trainParams.MBSize, grid);
      int NumLayers = netParams.Network.size();
      // initalize neural network (layers)
      for (int l = 0; l < (int)NumLayers; l++) {
        string networkType;
        if(l < (int)NumLayers-1) {
          networkType = "FullyConnected";
        } else {
          // Add a softmax layer to the end
          networkType = "softmax";
        }
        dnn->add(networkType, netParams.Network[l], trainParams.ActivationType, {new dropout(trainParams.DropOut)});
      }
    }

    if (grid.Rank() == 0) {
      cout << "Layer initialized:" << endl;
      if (g_AutoEncoder) {
        for (size_t n = 0; n < autoencoder->get_layers().size(); n++) {
          cout << "\tLayer[" << n << "]: " << autoencoder->get_layers()[n]->NumNeurons << endl;
        }
      } else {
        for (uint n = 0; n < dnn->get_layers().size(); n++) {
          cout << "\tLayer[" << n << "]: " << dnn->get_layers()[n]->NumNeurons << endl;
        }
      }
      cout << endl;
    }

    if (grid.Rank() == 0) {
      cout << "Parameter settings:" << endl;
      cout << "\tBlock size: " << perfParams.BlockSize << endl;
      cout << "\tEpochs: " << trainParams.EpochCount << endl;
      cout << "\tMini-batch size: " << trainParams.MBSize << endl;
      if(trainParams.MaxMBCount == 0) {
        cout << "\tMini-batch count (max): " << "unlimited" << endl;
      } else {
        cout << "\tMini-batch count (max): " << trainParams.MaxMBCount << endl;
      }
      cout << "\tLearning rate: " << trainParams.LearnRate << endl;
      cout << "\tEpoch count: " << trainParams.EpochCount << endl << endl;
      if(perfParams.MaxParIOSize == 0) {
        cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
      } else {
        cout << "\tMax Parallel I/O Fetch: " << perfParams.MaxParIOSize << endl;
      }
      cout << "\tDataset: " << trainParams.DatasetRootDir << endl;
    }

    // load parameters from file if available
    if (trainParams.LoadModel && trainParams.ParameterDir.length() > 0) {
      if (g_AutoEncoder) {
        autoencoder->load_from_file(trainParams.ParameterDir);
      } else {
        dnn->load_from_file(trainParams.ParameterDir);
      }
    }

    ///////////////////////////////////////////////////////////////////
    // load ImageNet label list file
    ///////////////////////////////////////////////////////////////////
#if 0
    CImageNet imagenet;
    if (!imagenet.loadList(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile,
                           trainParams.DatasetRootDir + g_ImageNet_TrainDir,
                           trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_ValLabelFile,
                           trainParams.DatasetRootDir + g_ImageNet_ValDir,
                           trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile,
                           trainParams.DatasetRootDir + g_ImageNet_TestDir)) {
      cout << "ImageNet list file error: " << grid.Rank() << endl;
      return -1;
    }
    if (grid.Rank() == 0) {
      cout << "ImageNet training/validating/testing list loaded: ";
      cout << imagenet.getNumTrainData() << ", ";
      cout << imagenet.getNumValData()   << ", ";
      cout << imagenet.getNumTestData()  << endl;
      cout << endl;
    }
    /* Limit the number to training data samples to the size of
       the data set or the user-specified maximum */
    int numTrainData;
    if(trainParams.MaxMBCount != 0 && trainParams.MaxMBCount * trainParams.MBSize < imagenet.getNumTrainData()) {
      numTrainData = trainParams.MaxMBCount * trainParams.MBSize;
    } else {
      numTrainData = imagenet.getNumTrainData();
    }
    int numValData;
    if(trainParams.MaxValidationSamples != 0 && trainParams.MaxValidationSamples < imagenet.getNumValData()) {
      numValData = trainParams.MaxValidationSamples;
    } else {
      numValData = imagenet.getNumValData();
    }
    int numTestData;
    if(trainParams.MaxTestSamples != 0 && trainParams.MaxTestSamples < imagenet.getNumTestData()) {
      numTestData = trainParams.MaxTestSamples;
    } else {
      numTestData = imagenet.getNumTestData();
    }
    int MBCount = numTrainData / trainParams.MBSize;
    if (grid.Rank() == 0) {
      cout << "Processing " << numTrainData << " ImageNet training images in " << MBCount << " batches." << endl;
    }
#endif
    mpi::Barrier(grid.Comm());


    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////

    int last_layer_size;
    if(g_AutoEncoder) {
      last_layer_size = netParams.Network[netParams.Network.size()-1]+1;
    } else {
      last_layer_size = netParams.Network[netParams.Network.size()-1];
    }

    // create a local matrix on each process for holding an input image
    Mat X_local(netParams.Network[0] + 1, trainParams.MBSize);
    Mat Y_local(last_layer_size, trainParams.MBSize);
    // create a distributed matrix on each process for input and output that stores the data on a single root node
    CircMat Xs(netParams.Network[0] + 1, trainParams.MBSize, grid);
    CircMat X(netParams.Network[0] + 1, 1, grid);
    CircMat XP(netParams.Network[0] + 1, 1, grid);

    CircMat Ys(last_layer_size, trainParams.MBSize, grid);
    CircMat Y(last_layer_size, 1, grid);
    CircMat YP(last_layer_size, 1, grid);

    vector<int> indices(numTrainData);

    // create a buffer for image data
    unsigned char *imagedata = new unsigned char[g_ImageNet_Width * g_ImageNet_Height * 3];

    //************************************************************************
    // read training state from checkpoint file if we have one
    //************************************************************************
    int epochStart; // epoch number we should start at
    int trainStart; // index into indices we should start at
    //bool restarted = restartShared(&epochStart, &trainStart, indices, trainParams, dnn);
    bool restarted = restartShared(&epochStart, &trainStart, indices, trainParams, dnn);
    if (! restarted) {
      // brand new run, set both starting values to 0
      epochStart = 0;
      trainStart = 0;

      // Note: libelemental intializes model params above with random values
      // seed the random number generator
      std::srand(trainParams.RandomSeed + 0);

      // Create a random ordering of the training set
      int tmpNumTrainData = numTrainData; //imagenet.getNumTrainData()-1;
      vector<int> trainingSet(tmpNumTrainData /*imagenet.getNumTrainData()-1*/);
      for (int n = 0; n < tmpNumTrainData/*imagenet.getNumTrainData()-1*/; n++) {
        trainingSet[n] = n;
      }
      if(trainParams.ShuffleTrainingData) {
        std::random_shuffle(trainingSet.begin(), trainingSet.end());
      }

      // select the first N from the randomly ordered training samples - initialize indices
      for (int n = 0; n < numTrainData; n++) {
        indices[n] = trainingSet[n];
      }
    }

    //************************************************************************
    // mainloop for train/validate
    //************************************************************************
    for (uint epoch = epochStart; epoch < trainParams.EpochCount; epoch++) {
      if (grid.Rank() == 0) {
        cout << "-----------------------------------------------------------" << endl;
        cout << "[" << epoch << "] Epoch (learning rate = " << trainParams.LearnRate << ")"<< endl;
        cout << "-----------------------------------------------------------" << endl;
      }

      if (!restarted && !g_AutoEncoder) {
        ((softmax_layer *)dnn->get_layers()[dnn->get_layers().size()-1])->resetCost();
        //              dnn->softmax->resetCost();
      }

      // TODO: need to save this in checkpoint?
      decayIterations = 1;

      //************************************************************************
      // training epoch loop
      //************************************************************************
      if (! restarted) {
        // randomly shuffle indices into training data at start of each epoch
        std::srand(trainParams.RandomSeed + epoch);
        std::random_shuffle(indices.begin(), indices.end());
      }

      // Determine how much parallel I/O
      int TargetMaxIOSize = 1;
      if (perfParams.MaxParIOSize > 0) {
        TargetMaxIOSize = (grid.Size() < perfParams.MaxParIOSize) ? grid.Size() : perfParams.MaxParIOSize;
      }

      // if (grid.Rank() == 0) {
      //   cout << "\rTraining:      " << endl; //flush;
      // }

      int trainOffset = trainStart;
      while (trainOffset < numTrainData) {
        Zero(X_local);
        Zero(Y_local);

        // assume each reader can fetch a whole minibatch of training data
        int trainBlock = TargetMaxIOSize * trainParams.MBSize;
        int trainRemaining = numTrainData - trainOffset;
        if (trainRemaining < trainBlock) {
          // not enough training data left for all readers to fetch a full batch
          // compute number of readers needed
          trainBlock = trainRemaining;
        }

        // How many parallel I/O streams can be fetched
        int IO_size = ceil((double)trainBlock / trainParams.MBSize);

        if (trainParams.EnableProfiling && grid.Rank() == 0) {
          timer_io.Start();
        }

        // read training data/label mini batch
        if (grid.Rank() < IO_size) {
          int myOffset = trainOffset + (grid.Rank() * trainParams.MBSize);
          int numImages = std::min(trainParams.MBSize, numTrainData - myOffset);
          getTrainDataMB(imagenet, &indices[myOffset], imagedata, X_local, Y_local, numImages, netParams.Network[0]);
        }
        mpi::Barrier(grid.Comm());

        if (grid.Rank() == 0) {
          if (trainParams.EnableProfiling) {
            sec_all_io += timer_io.Stop();
            timer_lbann.Start();
          }
          cout << "\rTraining: " << trainOffset << endl;
          //                  cout << "\b\b\b\b\b" << setw(5) << trainOffset << flush;
          //                  cout << "\r" << setw(5) << trainOffset << "\t" << std::flush;
          //                  cout << "\t" << setw(5) << trainOffset << "\t" << std::flush;
          //                  flush(cout);
#if 0
          {
            float progress = 0.0;
            while (progress < 1.0) {
              int barWidth = 70;

              std::cout << "[";
              int pos = barWidth * progress;
              for (int i = 0; i < barWidth; ++i) {
                if (i < pos) {
                  std::cout << "=";
                } else if (i == pos) {
                  std::cout << ">";
                } else {
                  std::cout << " ";
                }
              }
              std::cout << "] " << int(progress * 100.0) << " %\r";
              std::cout.flush();

              progress += 0.16; // for demonstration only
            }
            std::cout << std::endl;
          }
#endif
        }

        // train mini batch
        for(int r = 0; r < IO_size; r++) {
          Zero(Xs);
          Zero(Ys);
          Xs.SetRoot(r);
          Ys.SetRoot(r);
          //if (grid.Rank() == r) {
          //  Xs.CopyFromRoot(X_local);
          //  Ys.CopyFromRoot(Y_local);
          //}else {
          //  Xs.CopyFromNonRoot();
          //  Ys.CopyFromNonRoot();
          //}
          //mpi::Barrier(grid.Comm());
          if (grid.Rank() == r) {
            CopyFromRoot(X_local, Xs);
            CopyFromRoot(Y_local, Ys);
          } else {
            CopyFromNonRoot(Xs);
            CopyFromNonRoot(Ys);
          }


          if (g_AutoEncoder) {
            autoencoder->train(Xs, trainParams.LearnRate);
          } else {
            dnn->train(Xs, Ys, trainParams.LearnRate, trainParams.LearnRateMethod);
          }

#if 0
          if(/*n*/trainOffset + r * trainParams.MBSize > decayIterations * trainParams.LrDecayCycles) {
            trainParams.LearnRate = trainParams.LearnRate * trainParams.LrDecayRate;
            decayIterations++;
            if(grid.Rank() == 0) {
              cout << "Changing the learning rate to " << trainParams.LearnRate << " after processing " << (/*n*/trainOffset + r * trainParams.MBSize) << " dataums" << endl;
            }
          }
#endif
          mpi::Barrier(grid.Comm());
        }
        if (trainParams.EnableProfiling && grid.Rank() == 0) {
          sec_all_lbann += timer_lbann.Stop();
        }
        // Finished training on single pass of data
        mpi::Barrier(grid.Comm());

        // increment our offset into the training data
        trainOffset += trainBlock;

        //************************************************************************
        // checkpoint our training state
        //************************************************************************
        // TODO: checkpoint
        bool ckpt_epoch = ((trainOffset == numTrainData) && (trainParams.Checkpoint > 0) && (epoch % trainParams.Checkpoint == 0));
        if (trainParams.SaveModel && trainParams.ParameterDir.length() > 0 && ckpt_epoch) {
          checkpoint(epoch, trainOffset, indices, trainParams, dnn);
          checkpointShared(epoch, trainOffset, indices, trainParams, dnn);
        }
      }

      // reset our training offset for the next epoch
      restarted = false;
      trainStart = 0;

      if (grid.Rank() == 0) {
        cout << " ... done" << endl;
        if (trainParams.EnableProfiling) {
          double sec_all_total = sec_all_io + sec_all_lbann;

          double sec_mbatch_io = sec_all_io / (MBCount * (epoch+1));
          double sec_mbatch_lbann = sec_all_lbann / (MBCount * (epoch+1));
          double sec_mbatch_total = (sec_all_io + sec_all_lbann) / (MBCount * (epoch+1));

          double sec_each_io = sec_mbatch_io / trainParams.MBSize;
          double sec_each_lbann = sec_mbatch_lbann / trainParams.MBSize;
          double sec_each_total = (sec_mbatch_io + sec_mbatch_lbann) / trainParams.MBSize;

          if(!g_AutoEncoder) {
            double avg_cost = ((softmax_layer *)dnn->get_layers()[dnn->get_layers().size()-1])->avgCost();
            //                    double avg_cost = dnn->softmax->avgCost();
            cout << "Average softmax Cost: " << avg_cost << endl;
          }
          cout << "#, Host, Nodes, Processes, Cores, TasksPerNode, Epoch, Training Samples, Mini-Batch Size, Mini-Batch Count, Total Time, Total I/O, Total lbann, MB Time, MB I/O, MB lbann, Sample Time, Sample I/O, Sample lbann" << endl;
          cout << "# [RESULT], " << sysParams.HostName << ", " << sysParams.NumNodes << ", " << grid.Size() << ", " << sysParams.NumCores << ", " << sysParams.TasksPerNode << ", " << epoch << ", ";
          cout << numTrainData << ", " << trainParams.MBSize << ", " << MBCount << ", ";
          cout << sec_all_total    << ", " << sec_all_io    << ", " << sec_all_lbann    << ", ";
          cout << sec_mbatch_total << ", " << sec_mbatch_io << ", " << sec_mbatch_lbann << ", ";
          cout << sec_each_total   << ", " << sec_each_io   << ", " << sec_each_lbann   << endl;
#if 0
          cout << "Training time (sec): ";
          cout << "total: "      << sec_all_total    << " (I/O: " << sec_all_io    << ", lbann: " << sec_all_lbann    << ")" << endl;
          cout << "mini-batch: " << sec_mbatch_total << " (I/O: " << sec_mbatch_io << ", lbann: " << sec_mbatch_lbann << ")" << endl;
          cout << "each: "       << sec_each_total   << " (I/O: " << sec_each_io   << ", lbann: " << sec_each_lbann   << ")" << endl;
          cout << endl;
#endif
        }
      }

#if 1
      // Update the learning rate on each epoch
      trainParams.LearnRate = trainParams.LearnRate * trainParams.LrDecayRate;
      if(grid.Rank() == 0) {
        cout << "Changing the learning rate to " << trainParams.LearnRate << " after processing " << (epoch+1) << " epochs" << endl;
      }
#endif

      //************************************************************************
      // validating/testing loop
      //************************************************************************
      int numTopOneErrors = 0, numTopFiveErrors = 0;
      double sumerrors = 0;
      if (trainParams.EnableProfiling && grid.Rank() == 0) {
        timer_val.Start();
      }
      for (int n = 0; n < numValData; n++) {

        // read validating data/label
        int imagelabel;
        if (grid.Rank() == 0) {
          if(trainParams.TestWithTrainData) {
            getTrainData(imagenet, indices[n], imagedata, X, Y, netParams.Network[0]);
            for(int i = 0; i < Y.Height(); i++) {
              if(Y.GetLocal(i,0) == 1) {
                imagelabel = i;
              }
            }
          } else {
            getValData(imagenet, n, imagedata, X, imagelabel, netParams.Network[0]);
          }
        }
        mpi::Barrier(grid.Comm());

        if (g_AutoEncoder) {
          autoencoder->test(X, XP);

          // validate
          if (grid.Rank() == 0) {
            for (uint m = 0; m < netParams.Network[0]; m++) {
              sumerrors += ((X.GetLocal(m, 0) - XP.GetLocal(m, 0)) * (X.GetLocal(m, 0) - XP.GetLocal(m, 0)));
            }

            cout << "\rTesting: " << n;
          }
        } else {
          // test dnn
          dnn->test(X, Y);

          // validate
          if (grid.Rank() == 0) {
            int labelidx[5] = {-1, -1, -1, -1, -1};
            double labelmax[5] = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
            //                        cout << endl;
            for (int m = 0; m < netParams.Network[netParams.Network.size()-1]; m++) {
              for(int k = 0; k < 5; k++) {
                if (labelmax[k] <= Y.GetLocal(m, 0)) {
                  for(int i = 4; i > k; i--) {
                    labelmax[i] = labelmax[i-1];
                    labelidx[i] = labelidx[i-1];
                  }
                  labelmax[k] = Y.GetLocal(m, 0);
                  labelidx[k] = m;
                  break;
                }
              }
            }
            if (imagelabel != labelidx[0]) {
              numTopOneErrors++;
            }
            bool topFiveMatch = false;
            for(int i = 0; i < 5; i++) {
              if(imagelabel == labelidx[i]) {
                topFiveMatch = true;
                break;
              }
            }
            if(!topFiveMatch) {
              numTopFiveErrors++;
            }
            // Print(Y);
#if 0
            if(!topFiveMatch) {
              cout << "\rTesting: " << n << "th sample, " << numTopOneErrors << " top one errors and " << numTopFiveErrors
                   << " top five errors - image label " << imagelabel << " =?= ";
              for(int i = 0; i < 5; i++) {
                cout << labelidx[i] << "(" << labelmax[i] << ") ";
              }
              cout << endl;
              int bad_val = 0;
              for(int i = 0; i < Y.Height(); i++) {
                if(Y.GetLocal(i,0) < 0.00001) {
                  bad_val++;
                } else {
                  cout << i << "=" << Y.GetLocal(i,0) << " ";
                }
              }
              cout << endl;
              cout << bad_val << " insignificant values"<< endl << endl;
            }
#endif
          }
        }
      }
      if (grid.Rank() == 0) {
        if (trainParams.EnableProfiling) {
          sec_all_val += timer_val.Stop();
        }
        cout << endl;
        if (trainParams.EnableProfiling) {
          //                  double sec_all_vall_total = sec_all_io + sec_all_lbann;

          double sec_val_each_total = sec_all_val / (numValData * (epoch+1));

          cout << "Validation time (sec): ";
          cout << "total: "      << sec_all_val << endl;
          cout << "each: "       << sec_val_each_total << endl;
          cout << endl;
        }
      }

      if (g_AutoEncoder) {
        if (grid.Rank() == 0) {
          cout << "Sum. square errors: " << sumerrors << endl;
        }

        // save a couple of reconstructed outputs as image files
        int imagecount = sizeof(g_SaveImageIndex) / sizeof(int);
        uchar *pixels_gt = new uchar[netParams.Network[0] * imagecount];
        uchar *pixels_rc = new uchar[netParams.Network[0] * imagecount];

        for (int n = 0; n < imagecount; n++) {
          int imagelabel;
          if (grid.Rank() == 0) {
            if (1 || numValData <= 0) {
              getTrainData(imagenet, g_SaveImageIndex[n], imagedata, X, Y, netParams.Network[0]);
            } else {
              getValData(imagenet, g_SaveImageIndex[n], imagedata, X, imagelabel, netParams.Network[0]);
            }

            for (int y = 0; y < g_ImageNet_Height; y++)
              for (int x = 0; x < g_ImageNet_Width; x++)
                for (int ch = 0; ch < 3; ch++) {
                  pixels_gt[(y * g_ImageNet_Width * imagecount + x + g_ImageNet_Width * n) * 3 + ch] = imagedata[(y * g_ImageNet_Width + x) * 3 + ch];
                }
          }
          mpi::Barrier(grid.Comm());
          autoencoder->test(X, XP);

          if (grid.Rank() == 0) {
            for (uint m = 0; m < netParams.Network[0]; m++) {
              imagedata[m] = XP.GetLocal(m, 0) * 255;
            }

            for (int y = 0; y < g_ImageNet_Height; y++)
              for (int x = 0; x < g_ImageNet_Width; x++)
                for (int ch = 0; ch < 3; ch++) {
                  pixels_rc[(y * g_ImageNet_Width * imagecount + x + g_ImageNet_Width * n) * 3 + ch] = imagedata[(y * g_ImageNet_Width + x) * 3 + ch];
                }
          }
        }

        if (grid.Rank() == 0 && trainParams.SaveImageDir.length() > 0) {
          char imagepath_gt[512];
          char imagepath_rc[512];
          sprintf(imagepath_gt, "%s/lbann_autoencoder_imagenet_gt.png", trainParams.SaveImageDir.c_str());
          sprintf(imagepath_rc, "%s/lbann_autoencoder_imagenet_%04d.png", trainParams.SaveImageDir.c_str(), epoch);
          CImageUtil::savePNG(imagepath_gt, g_ImageNet_Width * imagecount, g_ImageNet_Height, true, pixels_gt);
          CImageUtil::savePNG(imagepath_rc, g_ImageNet_Width * imagecount, g_ImageNet_Height, true, pixels_rc);
        }

        delete [] pixels_gt;
        delete [] pixels_rc;
      } else {
        float topOneAccuracy = (float)(numValData - numTopOneErrors) / numValData * 100.0f;
        float topFiveAccuracy = (float)(numValData - numTopFiveErrors) / numValData * 100.0f;
        if (grid.Rank() == 0) {
          cout << "Top One Accuracy:  " << topOneAccuracy << "%" << endl;
          cout << "Top Five Accuracy: " << topFiveAccuracy << "%" << endl << endl;
        }
      }

      //************************************************************************
      // checkpoint our training state
      //************************************************************************
      /*
                  if (trainParams.SaveModel && trainParams.ParameterDir.length() > 0 &&
                      trainParams.Checkpoint > 0 && (epoch % trainParams.Checkpoint == 0))
                  {
                      checkpoint(epoch+1, trainParams, dnn);
                  }
      */
    }
    delete [] imagedata;

    // save final model parameters
    if (trainParams.SaveModel && trainParams.ParameterDir.length() > 0) {
      if (g_AutoEncoder) {
        autoencoder->save_to_file(trainParams.ParameterDir);
      } else {
        dnn->save_to_file(trainParams.ParameterDir);
      }
    }

    if (g_AutoEncoder) {
      delete autoencoder;
    } else {
      delete dnn;
    }
  } catch (exception& e) {
    ReportException(e);
  }

  // free all resources by El and MPI
  Finalize();

  return 0;
}
#endif
