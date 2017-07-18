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
// alexnet.cpp - AlexNet application for ImageNet classification
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/layers/regularizers/dropout.hpp"
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/data_readers/data_reader_imagenet_cv.hpp"
#include "lbann/data_readers/data_reader_imagenet_single_cv.hpp"

#include <time.h>
#include <sys/time.h>
#include <iomanip>
#include <string>

using namespace std;
using namespace lbann;
using namespace El;

#define PARTITIONED
#if defined(PARTITIONED)
#define DATA_LAYOUT data_layout::DATA_PARALLEL
#else
#define DATA_LAYOUT data_layout::MODEL_PARALLEL
#endif

void get_prev_neurons_and_index( lbann::sequential_model *model, int& prev_num_neurons, int& cur_index) {
  std::vector<Layer *>& layers = model->get_layers();
  prev_num_neurons = -1;
  if(layers.size() != 0) {
    Layer *prev_layer = layers.back();
    prev_num_neurons = prev_layer->get_num_neurons();
  }
  cur_index = layers.size();
}

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

int main(int argc, char *argv[]) {
  lbann_comm *comm = initialize(argc, argv, 42);

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

    bool unit_scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
    bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", true);
    bool unit_variance = Input("--unit-variance", "standardize to unit-variance", true);

    //if set to true, above three settings have no effect
    bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

    // Number of GPUs
    int num_gpus = Input("--num-gpus", "number of GPUs to use", -1);

    // Number of class labels
    int num_classes = Input("--num-classes", "number of class labels in dataset", 1000);

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

    // set up the normalizer
    std::unique_ptr<lbann::cv_normalizer> normalizer(new(lbann::cv_normalizer));
    normalizer->unit_scale(unit_scale);
    normalizer->subtract_mean(subtract_mean);
    normalizer->unit_variance(unit_variance);
    normalizer->z_score(z_score);

    // set up a custom transform (colorizer)
    std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));

    // set up the image preprocessor
    std::shared_ptr<cv_process> pp = std::make_shared<cv_process>();
    pp->set_normalizer(std::move(normalizer));
    pp->set_custom_transform2(std::move(colorizer));

    parallel_io = 1;

    std::map<execution_mode, generic_data_reader *> data_readers;
    ///////////////////////////////////////////////////////////////////
    // load training data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    if (not use_new_reader) {
      if (comm->am_world_master()) {
        cout << endl << "USING imagenet_reader_cv\n\n";
      }
      imagenet_reader_cv *imagenet_trainset = new imagenet_reader_cv(trainParams.MBSize, pp, true);
      imagenet_trainset->set_firstN(false);
      imagenet_trainset->set_role("train");
      imagenet_trainset->set_master(comm->am_world_master());
      imagenet_trainset->set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TrainDir);
      imagenet_trainset->set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile);
      imagenet_trainset->set_use_percent(trainParams.PercentageTrainingSamples);
      imagenet_trainset->set_validation_percent(trainParams.PercentageValidationSamples);
      imagenet_trainset->load();

      ///////////////////////////////////////////////////////////////////
      // create a validation set from the unused training data (ImageNet)
      ///////////////////////////////////////////////////////////////////
      imagenet_reader_cv *imagenet_validation_set = new imagenet_reader_cv(*imagenet_trainset); // Clone the training set object
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
      imagenet_reader_cv *imagenet_testset = new imagenet_reader_cv(trainParams.MBSize, pp, true);
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
      imagenet_readerSingle_cv *imagenet_trainset = new imagenet_readerSingle_cv(trainParams.MBSize, pp, true);
      imagenet_trainset->set_firstN(false);
      imagenet_trainset->set_role("train");
      imagenet_trainset->set_master(comm->am_world_master());
      imagenet_trainset->set_file_dir(trainParams.DatasetRootDir);

      stringstream ss;
      ss << "Single_" << g_ImageNet_TrainLabelFile.substr(0, g_ImageNet_TrainLabelFile.size()-4);
      imagenet_trainset->set_data_filename(ss.str());
      imagenet_trainset->set_validation_percent(trainParams.PercentageValidationSamples);

      imagenet_trainset->load();

      ///////////////////////////////////////////////////////////////////
      // create a validation set from the unused training data (ImageNet)
      ///////////////////////////////////////////////////////////////////
      imagenet_readerSingle_cv *imagenet_validation_set = new imagenet_readerSingle_cv(*imagenet_trainset); // Clone the training set object
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
      imagenet_readerSingle_cv *imagenet_testset = new imagenet_readerSingle_cv(trainParams.MBSize, pp, true);
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
   dnn->add_metric(new metrics::categorical_accuracy<DATA_LAYOUT>(comm));
    // input_layer *input_layer = new input_layer_distributed_minibatch(data_layout::DATA_PARALLEL, comm, trainParams.MBSize, data_readers);
#ifdef PARTITIONED
    input_layer *input_layer = new input_layer_partitioned_minibatch<>(comm, trainParams.MBSize, parallel_io, data_readers);
#else
    input_layer *input_layer = new input_layer_distributed_minibatch<DATA_LAYOUT>(comm, trainParams.MBSize, parallel_io, data_readers);
#endif
    dnn->add(input_layer);

    // Layer 1 (convolution)
    {
      optimizer *convolution_layer_optimizer = optimizer_fac->create_optimizer();
      int numDims = 2;
      int outputChannels = 96;
      int filterDims[] = {11, 11};
      int convPads[] = {0, 0};
      int convStrides[] = {4, 4};
      convolution_layer<> *layer
        = new convolution_layer<>(
          1,
          comm,
          trainParams.MBSize,
          numDims,
          outputChannels,
          filterDims,
          convPads,
          convStrides,
          weight_initialization::he_normal,
          convolution_layer_optimizer,
          true,
          cudnn);
      layer->set_l2_regularization_factor(0.0005);
      dnn->add(layer);
      Layer *relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          2,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(relu);
    }

    // Layer 2 (LRN)
    {
      int numDims = 2;
      int windowWidth = 5;
      DataType alpha = 0.0001;
      DataType beta = 0.75;
      DataType k = 2;
      local_response_normalization_layer<> *layer
        = new local_response_normalization_layer<>(
          3,
          comm,
          trainParams.MBSize,
          windowWidth,
          alpha,
          beta,
          k,
          cudnn);
      dnn->add(layer);
    }

    // Layer 3 (pooling)
    {
      int numDims = 2;
      int poolWindowDims[] = {3, 3};
      int poolPads[] = {0, 0};
      int poolStrides[] = {2, 2};
      pool_mode poolMode = pool_mode::max;
      pooling_layer<> *layer
        = new pooling_layer<>(
          4,
          comm,
          trainParams.MBSize,
          numDims,
          poolWindowDims,
          poolPads,
          poolStrides,
          poolMode,
          cudnn);
      dnn->add(layer);
    }

    // Layer 4 (convolution)
    {
      optimizer *convolution_layer_optimizer = optimizer_fac->create_optimizer();
      int numDims = 2;
      int outputChannels = 256;
      int filterDims[] = {5, 5};
      int convPads[] = {2, 2};
      int convStrides[] = {1, 1};
      convolution_layer<> *layer
        = new convolution_layer<>(
          5,
          comm,
          trainParams.MBSize,
          numDims,
          outputChannels,
          filterDims,
          convPads,
          convStrides,
          weight_initialization::he_normal,
          convolution_layer_optimizer,
          true,
          cudnn);
      layer->set_l2_regularization_factor(0.0005);
      dnn->add(layer);
      Layer *relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          6,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(relu);
    }

    // Layer 5 (LRN)
    {
      int numDims = 2;
      int windowWidth = 5;
      DataType alpha = 0.0001;
      DataType beta = 0.75;
      DataType k = 2;
      local_response_normalization_layer<> *layer
        = new local_response_normalization_layer<>(
          7,
          comm,
          trainParams.MBSize,
          windowWidth,
          alpha,
          beta,
          k,
          cudnn);
      dnn->add(layer);
    }

    // Layer 6 (pooling)
    {
      int numDims = 2;
      int poolWindowDims[] = {3, 3};
      int poolPads[] = {0, 0};
      int poolStrides[] = {2, 2};
      pool_mode poolMode = pool_mode::max;
      pooling_layer<> *layer
        = new pooling_layer<>(
          8,
          comm,
          trainParams.MBSize,
          numDims,
          poolWindowDims,
          poolPads,
          poolStrides,
          poolMode,
          cudnn);
      dnn->add(layer);
    }

    // Layer 7 (convolution)
    {
      optimizer *convolution_layer_optimizer = optimizer_fac->create_optimizer();
      int numDims = 2;
      int outputChannels = 384;
      int filterDims[] = {3, 3};
      int convPads[] = {1, 1};
      int convStrides[] = {1, 1};
      convolution_layer<> *layer
        = new convolution_layer<>(
          9,
          comm,
          trainParams.MBSize,
          numDims,
          outputChannels,
          filterDims,
          convPads,
          convStrides,
          weight_initialization::he_normal,
          convolution_layer_optimizer,
          true,
          cudnn);
      layer->set_l2_regularization_factor(0.0005);
      dnn->add(layer);
      Layer *relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          10,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(relu);
    }

    // Layer 8 (convolution)
    {
      optimizer *convolution_layer_optimizer = optimizer_fac->create_optimizer();
      int numDims = 2;
      int outputChannels = 384;
      int filterDims[] = {3, 3};
      int convPads[] = {1, 1};
      int convStrides[] = {1, 1};
      convolution_layer<> *layer
        = new convolution_layer<>(
          11,
          comm,
          trainParams.MBSize,
          numDims,
          outputChannels,
          filterDims,
          convPads,
          convStrides,
          weight_initialization::he_normal,
          convolution_layer_optimizer,
          true,
          cudnn);
      layer->set_l2_regularization_factor(0.0005);
      dnn->add(layer);
      Layer *relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          12,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(relu);
      dnn->add(layer);
     }

    // Layer 9 (convolution)
    {
      optimizer *convolution_layer_optimizer = optimizer_fac->create_optimizer();
      int numDims = 2;
      int outputChannels = 256;
      int filterDims[] = {3, 3};
      int convPads[] = {1, 1};
      int convStrides[] = {1, 1};
      convolution_layer<> *layer
        = new convolution_layer<>(
          13,
          comm,
          trainParams.MBSize,
          numDims,
          outputChannels,
          filterDims,
          convPads,
          convStrides,
          weight_initialization::he_normal,
          convolution_layer_optimizer,
          true,
          cudnn);
      layer->set_l2_regularization_factor(0.0005);
      dnn->add(layer);
      Layer *relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          14,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(relu);
    }

    // Layer 10 (pooling)
    {
      int numDims = 2;
      int poolWindowDims[] = {3, 3};
      int poolPads[] = {0, 0};
      int poolStrides[] = {2, 2};
      pool_mode poolMode = pool_mode::max;
      pooling_layer<> *layer
        = new pooling_layer<>(
          15,
          comm,
          trainParams.MBSize,
          numDims,
          poolWindowDims,
          poolPads,
          poolStrides,
          poolMode,
          cudnn);
      dnn->add(layer);
    }

    // Layer 11 (fully-connected)
    int prev_num_neurons;
    int layer_id;
    {
      //      get_prev_neurons_and_index( dnn, prev_num_neurons, layer_id);
      fully_connected_layer<DATA_LAYOUT> *fc 
        = new fully_connected_layer<DATA_LAYOUT>(
          16,
          comm,
          trainParams.MBSize,
          4096,
          weight_initialization::he_normal,
          dnn->create_optimizer());
      fc->set_l2_regularization_factor(0.0005);
      dnn->add(fc);
      Layer *relu
        = new relu_layer<DATA_LAYOUT>(
          17,
          comm,
          trainParams.MBSize);
      dnn->add(relu);
      Layer *dropout_layer
        = new dropout<DATA_LAYOUT>(
          18,
          comm,
          trainParams.MBSize,
          0.5);
      dnn->add(dropout_layer);
    }

    // Layer 12 (fully-connected)
    {
      //      get_prev_neurons_and_index( dnn, prev_num_neurons, layer_id);
      fully_connected_layer<DATA_LAYOUT> *fc 
        = new fully_connected_layer<DATA_LAYOUT>(
          19,
          comm,
          trainParams.MBSize,
          4096,
          weight_initialization::he_normal,
          dnn->create_optimizer(), 
          false);
      fc->set_l2_regularization_factor(0.0005);
      dnn->add(fc);
      Layer *relu
        = new relu_layer<DATA_LAYOUT>(
          20,
          comm,
          trainParams.MBSize);
      dnn->add(relu);
      Layer *dropout_layer
        = new dropout<DATA_LAYOUT>(
          21,
          comm,
          trainParams.MBSize,
          0.5);
      dnn->add(dropout_layer);
    }

    // Layer 13 (softmax)
    {
      // Fully-connected without bias before softmax.
      fully_connected_layer<DATA_LAYOUT> *fc
        = new fully_connected_layer<DATA_LAYOUT>(
          22,
          comm,
          trainParams.MBSize,
          1000,
          weight_initialization::he_normal,
          dnn->create_optimizer(),
          false);
      fc->set_l2_regularization_factor(0.0005);
      dnn->add(fc);
      //      get_prev_neurons_and_index( dnn, prev_num_neurons, layer_id);
      Layer *softmax 
        = new softmax_layer<DATA_LAYOUT>(
          23,
          comm,
          trainParams.MBSize,
          dnn->create_optimizer());
      dnn->add(softmax);
    }

    // target_layer *target_layer = new target_layer_distributed_minibatch(data_layout::MODEL_PARALLEL, comm, trainParams.MBSize, data_readers, true);
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
    lbann_callback_debug debug_cb(execution_mode::training);
    //    dnn->add_callback(&debug_cb);

    dnn->setup();

    if (grid.Rank() == 0) {
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

    ///////////////////////////////////////////////////////////////////
    // load ImageNet label list file
    ///////////////////////////////////////////////////////////////////
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
  finalize(comm);
  return 0;
}
