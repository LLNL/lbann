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
// dnn_multi_imagenet.cpp - DNN application for ImageNet with multiple models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/regularization/lbann_dropout.hpp"
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/data_readers/data_reader_imagenet_cv.hpp"

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
const int g_ImageNet_Width = 256;
const int g_ImageNet_Height = 256;

int main(int argc, char *argv[]) {
  // El initialization (similar to MPI_Init)
  Initialize(argc, argv);
  init_random(1);  // Deterministic initialization across every model.
  init_data_seq_random(1);
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
      static_cast<int>(lbann_callback_imcomm::ADAPTIVE_QUANTIZATION);
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

    // set up the normalizer
    std::unique_ptr<lbann::cv_normalizer> normalizer(new(lbann::cv_normalizer));
    normalizer->unit_scale(scale);
    normalizer->subtract_mean(subtract_mean);
    normalizer->unit_variance(unit_variance);
    normalizer->z_score(z_score);

    // set up a custom transform (colorizer)
    std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));

    // set up the image preprocessor
    std::shared_ptr<cv_process> pp = std::make_shared<cv_process>();
    pp->set_normalizer(std::move(normalizer));
    pp->set_custom_transform2(std::move(colorizer));

    ///////////////////////////////////////////////////////////////////
    // load training data (ImageNet)
    ///////////////////////////////////////////////////////////////////

    imagenet_reader_cv imagenet_trainset(trainParams.MBSize, pp, true);
    imagenet_trainset.set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TrainDir);
    imagenet_trainset.set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile);
    imagenet_trainset.set_validation_percent(trainParams.PercentageValidationSamples);
    imagenet_trainset.load();

    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    // Clone the training set object
    imagenet_reader_cv imagenet_validation_set(imagenet_trainset);
    // Swap the used and unused index sets so that it validates on the remaining data
    imagenet_validation_set.use_unused_index_set();

    if (comm->am_world_master()) {
      size_t num_train = imagenet_trainset.getNumData();
      size_t num_validate = imagenet_trainset.getNumData();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      cout << "Training using " << train_percent << "% of the training data set, which is " << imagenet_trainset.getNumData() << " samples." << endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << imagenet_validation_set.getNumData() << " samples." << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    imagenet_reader_cv imagenet_testset(trainParams.MBSize, pp, true);
    imagenet_testset.set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TestDir);
    imagenet_testset.set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile);
    imagenet_testset.set_use_percent(trainParams.PercentageTestingSamples);
    imagenet_testset.load();

    if (comm->am_world_master()) {
      cout << "Testing using " << (trainParams.PercentageTestingSamples*100) <<
           "% of the testing data set, which is " << imagenet_testset.getNumData() <<
           " samples." << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////

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

    layer_factory *lfac = new layer_factory();
    deep_neural_network dnn(trainParams.MBSize, comm,
                            new objective_functions::categorical_cross_entropy(comm), lfac, optimizer_fac);
    std::map<execution_mode, generic_data_reader *> data_readers = {
      std::make_pair(execution_mode::training,&imagenet_trainset),
      std::make_pair(execution_mode::validation, &imagenet_validation_set),
      std::make_pair(execution_mode::testing, &imagenet_testset)
    };
    //input_layer *input_layer = new input_layer_distributed_minibatch(comm, trainParams.MBSize, &imagenet_trainset, &imagenet_testset);
    input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, trainParams.MBSize, data_readers);
    dnn.add(input_layer);
    int NumLayers = netParams.Network.size();
    // initalize neural network (layers)
    std::unordered_set<int> layer_indices;
    for (int l = 0; l < NumLayers; l++) {
      int idx;
      if (l < (int)NumLayers-1) {
        idx = dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, netParams.Network[l],
                      trainParams.ActivationType,
                      weight_initialization::glorot_uniform,
        {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
      } else {
        // Add a softmax layer to the end
        idx = dnn.add("softmax", data_layout::MODEL_PARALLEL, netParams.Network[l],
                      activation_type::ID,
                      weight_initialization::glorot_uniform,
                      {});
      }
      layer_indices.insert(idx);
    }
    //target_layer *target_layer = new target_layer_distributed_minibatch(comm, trainParams.MBSize, &imagenet_trainset, &imagenet_testset, true);
    target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, trainParams.MBSize, data_readers, true);
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
  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);  /// Elemental exceptions
  }

  // free all resources by El and MPI
  Finalize();

  return 0;
}
