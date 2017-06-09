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
// lbann_alexnet.cpp - AlexNet application for ImageNet classification
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/regularization/lbann_l2_regularization.hpp"
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

using namespace std;
using namespace lbann;
using namespace El;

#define PARTITIONED

// train/test data info
const int g_SaveImageIndex[1] = {0}; // for auto encoder
//const int g_SaveImageIndex[5] = {293, 2138, 3014, 6697, 9111}; // for auto encoder
//const int g_SaveImageIndex[5] = {1000, 2000, 3000, 4000, 5000}; // for auto encoder
const string g_ImageNet_TrainDir = "resized_256x256/train/";
const string g_ImageNet_ValDir = "resized_256x256/val/";
const string g_ImageNet_TestDir = "resized_256x256/val/";
const string g_ImageNet_LabelDir = "labels/";
const uint g_ImageNet_Width = 256;
const uint g_ImageNet_Height = 256;

int main(int argc, char *argv[]) {
  // El initialization (similar to MPI_Init)
  Initialize(argc, argv);
  init_random(42);
  init_data_seq_random(42);
  lbann_comm *comm = NULL;

  try {
    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/ILSVRC2012/";
    trainParams.LearnRate = 5e-3;
    trainParams.DropOut = 0.5;
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

    // training settings
    int decayIterations = 1;

    bool scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
    bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", true);
    bool unit_variance = Input("--unit-variance", "standardize to unit-variance", true);

    //if set to true, above three settings have no effect
    bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

    // Number of GPUs
    Int num_gpus = Input("--num-gpus", "number of GPUs to use", -1);

    // Number of class labels
    Int num_classes = Input("--num-classes", "number of class labels in dataset", 1000);

    bool use_new_reader = Input("--new-reader", "use new data reader", false);

    ProcessInput();
    PrintInputReport();

    // set algorithmic blocksize
    SetBlocksize(perfParams.BlockSize);

    string g_ImageNet_TrainLabelFile;
    string g_ImageNet_ValLabelFile;
    string g_ImageNet_TestLabelFile;
    switch(num_classes) {
    case 10:
      g_ImageNet_TrainLabelFile = "train_c0-9.txt";
      g_ImageNet_ValLabelFile   = "val_c0-9.txt";
      g_ImageNet_TestLabelFile  = "val_c0-9.txt";
      break;
    case 100:
      g_ImageNet_TrainLabelFile = "train_c0-99.txt";
      g_ImageNet_ValLabelFile   = "val_c0-99.txt";
      g_ImageNet_TestLabelFile  = "val_c0-99.txt";
      break;
    case 300:
      g_ImageNet_TrainLabelFile = "train_c0-299.txt";
      g_ImageNet_ValLabelFile   = "val_c0-299.txt";
      g_ImageNet_TestLabelFile  = "val_c0-299.txt";
      break;
    default:
      g_ImageNet_TrainLabelFile = "train.txt";
      g_ImageNet_ValLabelFile   = "val.txt";
      g_ImageNet_TestLabelFile  = "val.txt";
    }
    if (comm->am_world_master()) {
      cout << "Train set label file: " << g_ImageNet_TrainLabelFile << "\n"
           << "Validation set label file: " << g_ImageNet_ValLabelFile << "\n"
           << "Test set label file: " << g_ImageNet_TestLabelFile << "\n";
    }

    // create timer for performance measurement
    Timer timer_io;
    Timer timer_lbann;
    Timer timer_val;
    double sec_all_io = 0;
    double sec_all_lbann = 0;
    double sec_all_val = 0;

    // Set up the communicator and get the grid.
    comm = new lbann_comm(trainParams.ProcsPerModel);
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
      imagenet_trainset->set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TrainDir);
      imagenet_trainset->set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile);
      imagenet_trainset->set_use_percent(trainParams.PercentageTrainingSamples);
      imagenet_trainset->set_validation_percent(trainParams.PercentageValidationSamples);
      imagenet_trainset->load();

      imagenet_trainset->scale(scale);
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
      imagenet_testset->set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TestDir);
      imagenet_testset->set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile);
      imagenet_testset->set_use_percent(trainParams.PercentageTestingSamples);
      imagenet_testset->load();

      if (comm->am_world_master()) {
        cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << imagenet_testset->getNumData() << " samples." << endl;
      }

      imagenet_testset->scale(scale);
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
      imagenet_trainset->set_file_dir(trainParams.DatasetRootDir);

      stringstream ss;
      ss << "Single_" << g_ImageNet_TrainLabelFile.substr(0, g_ImageNet_TrainLabelFile.size()-4);
      imagenet_trainset->set_data_filename(ss.str());
      imagenet_trainset->set_validation_percent(trainParams.PercentageValidationSamples);

      imagenet_trainset->load();

      imagenet_trainset->scale(scale);
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
      imagenet_testset->set_file_dir(trainParams.DatasetRootDir);
      imagenet_testset->set_data_filename(ss.str());
      imagenet_testset->set_use_percent(trainParams.PercentageTestingSamples);
      imagenet_testset->load();

      if (comm->am_world_master()) {
        cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << imagenet_testset->getNumData() << " samples." << endl;
      }

      imagenet_testset->scale(scale);
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
      optimizer_fac = new sgd_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, false);
    }

    // Initialize cuDNN (if detected)
#if __LIB_CUDNN
    cudnn::cudnn_manager *cudnn = new cudnn::cudnn_manager(comm, num_gpus);
#else // __LIB_CUDNN
    cudnn::cudnn_manager *cudnn = NULL;
#endif // __LIB_CUDNN

    deep_neural_network *dnn = NULL;
   dnn = new deep_neural_network(trainParams.MBSize, comm, new objective_functions::categorical_cross_entropy(comm), optimizer_fac);
    dnn->add_metric(new metrics::categorical_accuracy(data_layout::DATA_PARALLEL, comm));
    // input_layer *input_layer = new input_layer_distributed_minibatch(data_layout::DATA_PARALLEL, comm, (int) trainParams.MBSize, data_readers);
#ifdef PARTITIONED
    input_layer *input_layer = new input_layer_partitioned_minibatch_parallel_io<data_layout::DATA_PARALLEL>(comm, parallel_io, (int) trainParams.MBSize, data_readers);
#else
    input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(data_layout::DATA_PARALLEL, comm, parallel_io, (int) trainParams.MBSize, data_readers);
#endif
    dnn->add(input_layer);

    // Layer 1 (convolutional)
    {
      optimizer *convolution_layer_optimizer = optimizer_fac->create_optimizer();
      Int numDims = 2;
      Int inputChannels = 3;
      Int inputDims[] = {256, 256};
      Int outputChannels = 96;
      Int filterDims[] = {11, 11};
      Int convPads[] = {0, 0};
      Int convStrides[] = {4, 4};
      convolutional_layer *layer
        = new convolutional_layer(1, numDims, inputChannels, inputDims,
                                  outputChannels, filterDims,
                                  convPads, convStrides,
                                  trainParams.MBSize,
                                  activation_type::RELU,
                                  weight_initialization::he_normal,
                                  comm, convolution_layer_optimizer,
      {new l2_regularization(0.0005)},
      cudnn);
      dnn->add(layer);
    }

    // Layer 2 (LRN)
    {
      int numDims = 2;
      int channels = 96;
      int dims[] = {62, 62};
      Int windowWidth = 5;
      DataType alpha = 0.0001;
      DataType beta = 0.75;
      DataType k = 2;
      local_response_normalization_layer *layer
        = new local_response_normalization_layer(2, numDims, channels, dims,
            windowWidth, alpha, beta, k,
            trainParams.MBSize, comm, cudnn);
      dnn->add(layer);
    }

    // Layer 3 (pooling)
    {
      int numDims = 2;
      int channels = 96;
      int inputDim[] = {62, 62};
      int poolWindowDims[] = {3, 3};
      int poolPads[] = {0, 0};
      int poolStrides[] = {2, 2};
      pool_mode poolMode = pool_mode::max;
      pooling_layer *layer
        = new pooling_layer(3, numDims, channels, inputDim,
                            poolWindowDims, poolPads, poolStrides, poolMode,
                            trainParams.MBSize,
                            comm,
                            cudnn);
      dnn->add(layer);
    }

    // Layer 4 (convolutional)
    {
      optimizer *convolution_layer_optimizer = optimizer_fac->create_optimizer();
      Int numDims = 2;
      Int inputChannels = 96;
      Int inputDims[] = {30, 30};
      Int outputChannels = 256;
      Int filterDims[] = {5, 5};
      Int convPads[] = {2, 2};
      Int convStrides[] = {1, 1};
      convolutional_layer *layer
        = new convolutional_layer(4, numDims, inputChannels, inputDims,
                                  outputChannels, filterDims,
                                  convPads, convStrides,
                                  trainParams.MBSize,
                                  activation_type::RELU,
                                  weight_initialization::he_normal,
                                  comm, convolution_layer_optimizer,
      {new l2_regularization(0.0005)},
      cudnn);
      dnn->add(layer);
    }

    // Layer 5 (LRN)
    {
      int numDims = 2;
      int channels = 256;
      int dims[] = {30, 30};
      Int windowWidth = 5;
      DataType alpha = 0.0001;
      DataType beta = 0.75;
      DataType k = 2;
      local_response_normalization_layer *layer
        = new local_response_normalization_layer(5, numDims, channels, dims,
            windowWidth, alpha, beta, k,
            trainParams.MBSize, comm, cudnn);
      dnn->add(layer);
    }

    // Layer 6 (pooling)
    {
      int numDims = 2;
      int channels = 256;
      int inputDim[] = {30, 30};
      int poolWindowDims[] = {3, 3};
      int poolPads[] = {0, 0};
      int poolStrides[] = {2, 2};
      pool_mode poolMode = pool_mode::max;
      pooling_layer *layer
        = new pooling_layer(6, numDims, channels, inputDim,
                            poolWindowDims, poolPads, poolStrides, poolMode,
                            trainParams.MBSize,
                            comm,
                            cudnn);
      dnn->add(layer);
    }

    // Layer 7 (convolutional)
    {
      optimizer *convolution_layer_optimizer = optimizer_fac->create_optimizer();
      Int numDims = 2;
      Int inputChannels = 256;
      Int inputDims[] = {14, 14};
      Int outputChannels = 384;
      Int filterDims[] = {3, 3};
      Int convPads[] = {1, 1};
      Int convStrides[] = {1, 1};
      convolutional_layer *layer
        = new convolutional_layer(7, numDims, inputChannels, inputDims,
                                  outputChannels, filterDims,
                                  convPads, convStrides,
                                  trainParams.MBSize,
                                  activation_type::RELU,
                                  weight_initialization::he_normal,
                                  comm, convolution_layer_optimizer,
      {new l2_regularization(0.0005)},
      cudnn);
      dnn->add(layer);
    }

    // Layer 8 (convolutional)
    {
      optimizer *convolution_layer_optimizer = optimizer_fac->create_optimizer();
      Int numDims = 2;
      Int inputChannels = 384;
      Int inputDims[] = {14, 14};
      Int outputChannels = 384;
      Int filterDims[] = {3, 3};
      Int convPads[] = {1, 1};
      Int convStrides[] = {1, 1};
      convolutional_layer *layer
        = new convolutional_layer(8, numDims, inputChannels, inputDims,
                                  outputChannels, filterDims,
                                  convPads, convStrides,
                                  trainParams.MBSize,
                                  activation_type::RELU,
                                  weight_initialization::he_normal,
                                  comm, convolution_layer_optimizer,
      {new l2_regularization(0.0005)},
      cudnn);
      dnn->add(layer);
    }

    // Layer 9 (convolutional)
    {
      optimizer *convolution_layer_optimizer = optimizer_fac->create_optimizer();
      Int numDims = 2;
      Int inputChannels = 384;
      Int inputDims[] = {14, 14};
      Int outputChannels = 256;
      Int filterDims[] = {3, 3};
      Int convPads[] = {1, 1};
      Int convStrides[] = {1, 1};
      convolutional_layer *layer
        = new convolutional_layer(9, numDims, inputChannels, inputDims,
                                  outputChannels, filterDims,
                                  convPads, convStrides,
                                  trainParams.MBSize,
                                  activation_type::RELU,
                                  weight_initialization::he_normal,
                                  comm, convolution_layer_optimizer,
      {new l2_regularization(0.0005)},
      cudnn);
      dnn->add(layer);
    }

    // Layer 10 (pooling)
    {
      int numDims = 2;
      int channels = 256;
      int inputDim[] = {14, 14};
      int poolWindowDims[] = {3, 3};
      int poolPads[] = {0, 0};
      int poolStrides[] = {2, 2};
      pool_mode poolMode = pool_mode::max;
      pooling_layer *layer
        = new pooling_layer(10, numDims, channels, inputDim,
                            poolWindowDims, poolPads, poolStrides, poolMode,
                            trainParams.MBSize,
                            comm,
                            cudnn);
      dnn->add(layer);
    }

    // Layer 11 (fully-connected)
    dnn->add("FullyConnected",
             data_layout::MODEL_PARALLEL,
             4096,
             activation_type::RELU,
    weight_initialization::he_normal, {
      new dropout(data_layout::MODEL_PARALLEL, comm, 0.5),
      new l2_regularization(0.0005)
    });

    // Layer 12 (fully-connected)
    dnn->add("FullyConnected",
             data_layout::MODEL_PARALLEL,
             4096,
             activation_type::RELU,
    weight_initialization::he_normal, {
      new dropout(data_layout::MODEL_PARALLEL, comm, 0.5),
      new l2_regularization(0.0005)
    });

    // Layer 13 (softmax)
    dnn->add("Softmax",
             data_layout::MODEL_PARALLEL,
             1000,
             activation_type::ID,
             weight_initialization::he_normal,
    {new l2_regularization(0.0005)});

    // target_layer *target_layer = new target_layer_distributed_minibatch(data_layout::MODEL_PARALLEL, comm, (int) trainParams.MBSize, data_readers, true);
#ifdef PARTITIONED
    target_layer *target_layer = new target_layer_partitioned_minibatch_parallel_io<data_layout::MODEL_PARALLEL>(comm, parallel_io, (int) trainParams.MBSize, data_readers, true);
#else
    target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, (int) trainParams.MBSize, data_readers, true);
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

    dnn->setup();

    if (grid.Rank() == 0) {
      cout << "Layer initialized:" << endl;
      for (uint n = 0; n < dnn->get_layers().size(); n++) {
        cout << "\tLayer[" << n << "]: " << dnn->get_layers()[n]->NumNeurons << endl;
      }
      cout << endl;
    }

    if (grid.Rank() == 0) {
      cout << "Parameter settings:" << endl;
      cout << "\tBlock size: " << perfParams.BlockSize << endl;
      cout << "\tEpochs: " << trainParams.EpochCount << endl;
      cout << "\tMini-batch size: " << trainParams.MBSize << endl;
      // if(trainParams.MaxMBCount == 0) {
      //   cout << "\tMini-batch count (max): " << "unlimited" << endl;
      // }else {
      //   cout << "\tMini-batch count (max): " << trainParams.MaxMBCount << endl;
      // }
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
    last_layer_size = netParams.Network[netParams.Network.size()-1];

    //************************************************************************
    // read training state from checkpoint file if we have one
    //************************************************************************
    int epochStart = 0; // epoch number we should start at
    int trainStart; // index into indices we should start at

    //************************************************************************
    // mainloop for train/validate
    //************************************************************************
    for (int epoch = epochStart; epoch < trainParams.EpochCount; epoch++) {

      // TODO: need to save this in checkpoint?
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
  Finalize();

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
    if (!imagenet_trainset->load(trainParams.DatasetRootDir, g_MNIST_TrainImageFile, g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile)) {
      if (comm->am_world_master()) {
        cout << "ImageNet train data error" << endl;
      }
      return -1;
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    mnist_reader imagenet_testset(trainParams.MBSize, true, grid.Rank()*trainParams.MBSize, parallel_io*trainParams.MBSize);
    if (!imagenet_testset->load(g_MNIST_Dir, g_MNIST_TestImageFile, g_MNIST_TestLabelFile)) {
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
      optimizer_fac = new sgd_factory(grid, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
    }

    deep_neural_network *dnn = NULL;
    {
      dnn = new deep_neural_network(optimizer_fac, trainParams.MBSize, grid);
      int NumLayers = netParams.Network.size();
      // initalize neural network (layers)
      for (int l = 0; l < (int)NumLayers; l++) {
        string networkType;
        if(l < (int)NumLayers-1) {
          networkType = "FullyConnected";
        } else {
          // Add a softmax layer to the end
          networkType = "Softmax";
        }
        dnn->add(networkType, netParams.Network[l], trainParams.ActivationType, {new dropout(trainParams.DropOut)});
      }
    }

    if (grid.Rank() == 0) {
      cout << "Layer initialized:" << endl;
      for (uint n = 0; n < dnn->get_layers().size(); n++) {
        cout << "\tLayer[" << n << "]: " << dnn->get_layers()[n]->NumNeurons << endl;
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
      dnn->load_from_file(trainParams.ParameterDir);
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
    last_layer_size = netParams.Network[netParams.Network.size()-1];

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
    for (int epoch = epochStart; epoch < trainParams.EpochCount; epoch++) {
      if (grid.Rank() == 0) {
        cout << "-----------------------------------------------------------" << endl;
        cout << "[" << epoch << "] Epoch (learning rate = " << trainParams.LearnRate << ")"<< endl;
        cout << "-----------------------------------------------------------" << endl;
      }

      if (!restarted) {
        ((SoftmaxLayer *)dnn->get_layers()[dnn->get_layers().size()-1])->resetCost();
        //              dnn->Softmax->resetCost();
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


          dnn->train(Xs, Ys, trainParams.LearnRate, trainParams.LearnRateMethod);

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

          double avg_cost = ((SoftmaxLayer *)dnn->get_layers()[dnn->get_layers().size()-1])->avgCost();
          //                    double avg_cost = dnn->Softmax->avgCost();
          cout << "Average Softmax Cost: " << avg_cost << endl;
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

        {
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

      float topOneAccuracy = (float)(numValData - numTopOneErrors) / numValData * 100.0f;
      float topFiveAccuracy = (float)(numValData - numTopFiveErrors) / numValData * 100.0f;
      if (grid.Rank() == 0) {
        cout << "Top One Accuracy:  " << topOneAccuracy << "%" << endl;
        cout << "Top Five Accuracy: " << topFiveAccuracy << "%" << endl << endl;
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
      dnn->save_to_file(trainParams.ParameterDir);
    }

    delete dnn;
  } catch (exception& e) {
    ReportException(e);
  }

  // free all resources by El and MPI
  Finalize();

  return 0;
}
#endif
