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
// lbann_dnn_multi_imagenet.cpp - DNN application for ImageNet with multiple models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/regularization/lbann_dropout.hpp"
#include "lbann/data_readers/lbann_image_utils.hpp"

#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <iomanip>
#include <string>

//#include <algorithm>
//#include <random>

using namespace std;
using namespace lbann;
using namespace El;


// train/test data info
const int g_SaveImageIndex[1] = {0}; // for auto encoder
//const int g_SaveImageIndex[5] = {293, 2138, 3014, 6697, 9111}; // for auto encoder
//const int g_SaveImageIndex[5] = {1000, 2000, 3000, 4000, 5000}; // for auto encoder
const string g_ImageNet_TrainDir = "resized_256x256/train/";
const string g_ImageNet_ValDir = "resized_256x256/val/";
const string g_ImageNet_TestDir = "resized_256x256/val/"; //test/";
const string g_ImageNet_LabelDir = "labels/";
const string g_ImageNet_TrainLabelFile = "train_c0-9.txt";
const string g_ImageNet_ValLabelFile = "val.txt";
const string g_ImageNet_TestLabelFile = "val_c0-9.txt"; //"test.txt";
const uint g_ImageNet_Width = 256;
const uint g_ImageNet_Height = 256;

int main(int argc, char* argv[])
{
  // El initialization (similar to MPI_Init)
  Initialize(argc, argv);
  init_random(1);  // Deterministic initialization across every model.
  lbann_comm *comm = NULL;

  try {
    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/ILSVRC2012/";
    trainParams.EpochCount = 20;
    trainParams.DropOut = -1.0f;
    trainParams.ProcsPerModel = 48;
    trainParams.IntermodelCommMethod =
      static_cast<int>(lbann_callback_imcomm::ADAPTIVE_THRESH_QUANTIZATION);
    trainParams.parse_params();
    trainParams.PercentageTrainingSamples = 0.80;
    trainParams.PercentageValidationSamples = 1.00;
    PerformanceParams perfParams;
    perfParams.parse_params();
    // Read in the user specified network topology
    NetworkParams netParams;
    netParams.parse_params();
    // Get some environment variables from the launch
    SystemParams sysParams;
    sysParams.parse_params();

    // training settings
    bool scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
    bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", false);
    bool unit_variance = Input("--unit-variance", "standardize to unit-variance", false);

    //if set to true, above three settings have no effect
    bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

    ProcessInput();
    PrintInputReport();

    // set algorithmic blocksize
    SetBlocksize(perfParams.BlockSize);

    // Set up the communicator and get the grid.
    comm = new lbann_comm(trainParams.ProcsPerModel);
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
    parallel_io = 1;
    
    ///////////////////////////////////////////////////////////////////
    // load training data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    DataReader_ImageNet imagenet_trainset(trainParams.MBSize, true);
    bool training_set_loaded = false;
    training_set_loaded = imagenet_trainset.load(
      trainParams.DatasetRootDir + g_ImageNet_TrainDir, 
      trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile,
      trainParams.PercentageTrainingSamples);
    if (!training_set_loaded) {
      if (comm->am_world_master()) {
        cerr << __FILE__ << " " << __LINE__ <<
          " ImageNet train data error: training set was not loaded" << endl;
      }
      return -1;
    }
    if (comm->am_world_master()) {
      cout << "Training using " << (trainParams.PercentageTrainingSamples*100) <<
        "% of the training data set, which is " << imagenet_trainset.getNumData() <<
        " samples." << endl;
    }

    imagenet_trainset.scale(scale);
    imagenet_trainset.subtract_mean(subtract_mean);
    imagenet_trainset.unit_variance(unit_variance);
    imagenet_trainset.z_score(z_score);

    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    // Clone the training set object
    DataReader_ImageNet imagenet_validation_set(imagenet_trainset);
    // Swap the used and unused index sets so that it validates on the remaining data
    if (!imagenet_validation_set.swap_used_and_unused_index_sets()) {
      if (comm->am_world_master()) {
        cerr << __FILE__ << " " << __LINE__ << " ImageNet validation data error" << endl;
      }
      return -1;
    }

    if (trainParams.PercentageValidationSamples == 1.00) {
      if (comm->am_world_master()) {
        cout << "Validating training using " <<
          ((1.00 - trainParams.PercentageTrainingSamples)*100) <<
          "% of the training data set, which is " <<
          imagenet_validation_set.getNumData() << " samples." << endl;
      }
    } else {
      size_t preliminary_validation_set_size = imagenet_validation_set.getNumData();
      size_t final_validation_set_size = imagenet_validation_set.trim_data_set(trainParams.PercentageValidationSamples);
      if (comm->am_world_master()) {
        cout << "Trim the validation data set from " <<
          preliminary_validation_set_size << " samples to " <<
          final_validation_set_size << " samples." << endl;
      }
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    DataReader_ImageNet imagenet_testset(trainParams.MBSize, true);
    bool testing_set_loaded = false;
    testing_set_loaded = imagenet_testset.load(
      trainParams.DatasetRootDir + g_ImageNet_TestDir,  
      trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile, 
      trainParams.PercentageTestingSamples);
    if (!testing_set_loaded) {
      if (comm->am_world_master()) {
        cerr << __FILE__ << " " << __LINE__ <<
          " ImageNet Test data error: testing set was not loaded" << endl;
      }
      return -1;
    }
    if (comm->am_world_master()) {
      cout << "Testing using " << (trainParams.PercentageTestingSamples*100) <<
        "% of the testing data set, which is " << imagenet_testset.getNumData() <<
        " samples." << endl;
    }

    imagenet_testset.scale(scale);
    imagenet_testset.subtract_mean(subtract_mean);
    imagenet_testset.unit_variance(unit_variance);
    imagenet_testset.z_score(z_score);


    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////
    Optimizer_factory *optimizer;
    if (trainParams.LearnRateMethod == 1) { // Adagrad
      optimizer = new Adagrad_factory(comm, trainParams.LearnRate);
    } else if (trainParams.LearnRateMethod == 2) { // RMSprop
      optimizer = new RMSprop_factory(comm/*, trainParams.LearnRate*/);
    } else if (trainParams.LearnRateMethod == 3) { // Adam
      optimizer = new Adam_factory(comm, trainParams.LearnRate);
    } else {
      optimizer = new SGD_factory(comm, trainParams.LearnRate, 0.9,
                                  trainParams.LrDecayRate, true);
    }

    layer_factory* lfac = new layer_factory();
    deep_neural_network dnn(trainParams.MBSize, comm,
                            new objective_functions::categorical_cross_entropy(comm), lfac, optimizer);
    std::map<execution_mode, DataReader*> data_readers = {
      std::make_pair(execution_mode::training,&imagenet_trainset), 
      std::make_pair(execution_mode::validation, &imagenet_validation_set), 
      std::make_pair(execution_mode::testing, &imagenet_testset)
    };
    //input_layer *input_layer = new input_layer_distributed_minibatch(comm, (int) trainParams.MBSize, &imagenet_trainset, &imagenet_testset);
    input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, data_readers);
    dnn.add(input_layer);
    int NumLayers = netParams.Network.size();
    // initalize neural network (layers)
    std::unordered_set<uint> layer_indices;
    for (int l = 0; l < (int)NumLayers; l++) {
      uint idx;
      if (l < (int)NumLayers-1) {
        idx = dnn.add("FullyConnected", netParams.Network[l],
                      trainParams.ActivationType,
                      weight_initialization::glorot_uniform,
                      {new dropout(comm, trainParams.DropOut)});
      } else {
        // Add a softmax layer to the end
        idx = dnn.add("Softmax", netParams.Network[l],
                      activation_type::ID,
                      weight_initialization::glorot_uniform,
                      {});
      }
      layer_indices.insert(idx);
    }
    //target_layer *target_layer = new target_layer_distributed_minibatch(comm, (int) trainParams.MBSize, &imagenet_trainset, &imagenet_testset, true);
    target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, data_readers, true);
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
    // lbann_callback_io io_cb({0});
    // dnn->add_callback(&io_cb);
    // Do global inter-model updates.
    lbann_callback_imcomm imcomm_cb(
      static_cast<lbann_callback_imcomm::comm_type>(
        trainParams.IntermodelCommMethod),
      layer_indices, &summarizer);

    if (comm->am_world_master()) {
      cout << "Layer initialized:" << endl;
      for (uint n = 0; n < dnn.get_layers().size(); n++) {
        cout << "\tLayer[" << n << "]: " << dnn.get_layers()[n]->NumNeurons << endl;
      }
      cout << endl;
    }

    if (comm->am_world_master()) {
      cout << "Parameter settings:" << endl;
      cout << "\tBlock size: " << perfParams.BlockSize << endl;
      cout << "\tEpochs: " << trainParams.EpochCount << endl;
      cout << "\tMini-batch size: " << trainParams.MBSize << endl;
      cout << "\tLearning rate: " << trainParams.LearnRate << endl;
      cout << "\tEpoch count: " << trainParams.EpochCount << endl << endl;
      cout << "\tDataset: " << trainParams.DatasetRootDir << endl;
    }

    comm->global_barrier();

    // Initialize model.
    dnn.setup();
    
    comm->global_barrier();

    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////
    for (int t = 0; t < trainParams.EpochCount; ++t) {
      dnn.train(1, true);
      dnn.evaluate();
    }
  }
  catch (lbann_exception& e) { lbann_report_exception(e, comm); }
  catch (exception& e) { ReportException(e); } /// Elemental exceptions

  // free all resources by El and MPI
  Finalize();

  return 0;
}
