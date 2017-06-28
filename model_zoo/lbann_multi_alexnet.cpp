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
#include "lbann/data_readers/lbann_image_utils.hpp"

using namespace lbann;

// train/test data info
const string g_ImageNet_TrainDir = "resized_256x256/train/";
const string g_ImageNet_ValDir = "resized_256x256/val/";
const string g_ImageNet_TestDir = "resized_256x256/val/";
const string g_ImageNet_LabelDir = "labels/";
const int g_ImageNet_Width = 256;
const int g_ImageNet_Height = 256;




// #define PARTITIONED

int main(int argc, char *argv[]) {
  lbann_comm *comm = initialize(argc, argv, 42);

  try {
    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/ILSVRC2012/";
    trainParams.LearnRate = 1e-2;
    trainParams.DropOut = 0.5;
    trainParams.ProcsPerModel = 4;
    trainParams.IntermodelCommMethod
      = static_cast<int>(lbann_callback_imcomm::NORMAL);
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
    bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", true);
    bool unit_variance = Input("--unit-variance", "standardize to unit-variance", true);

    //if set to true, above three settings have no effect
    bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

    // Number of GPUs
#if __LIB_CUDNN
    int num_gpus = Input("--num-gpus", "number of GPUs to use", -1);
#endif

    // Number of class labels
    int num_classes = Input("--num-classes", "number of class labels in dataset", 1000);

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

    // Set up the communicator and get the grid.
    comm->split_models(trainParams.ProcsPerModel);
    Grid& grid = comm->get_model_grid();
    if (comm->am_world_master()) {
      std::cout << "Number of models: " << comm->get_num_models() << std::endl;
      std::cout << "Grid is " << grid.Height() << " x " << grid.Width() << std::endl;
      std::cout << std::endl;
    }

    int parallel_io = perfParams.MaxParIOSize;
    if(parallel_io == 0) {
      if(comm->am_world_master()) {
        std::cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() << " (Limited to # Processes)" << std::endl;
      }
      parallel_io = comm->get_procs_per_model();
    } else {
      if(comm->am_world_master()) {
        std::cout << "\tMax Parallel I/O Fetch: " << parallel_io << std::endl;
      }
    }

    ///////////////////////////////////////////////////////////////////
    // load training data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    imagenet_reader imagenet_trainset(trainParams.MBSize, true);
    imagenet_trainset.set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TrainDir);
    imagenet_trainset.set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile);
    imagenet_trainset.set_use_percent(trainParams.PercentageTrainingSamples);
    imagenet_trainset.set_validation_percent(trainParams.PercentageValidationSamples);
    imagenet_trainset.load();

    imagenet_trainset.scale(scale);
    imagenet_trainset.subtract_mean(subtract_mean);
    imagenet_trainset.unit_variance(unit_variance);
    imagenet_trainset.z_score(z_score);

    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    imagenet_reader imagenet_validation_set(imagenet_trainset); // Clone the training set object
    imagenet_validation_set.use_unused_index_set();

    if (comm->am_world_master()) {
      size_t num_train = imagenet_trainset.getNumData();
      size_t num_validate = imagenet_validation_set.getNumData();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      std::cout << "Training using " << train_percent << "% of the training data set, which is " << imagenet_trainset.getNumData() << " samples." << std::endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << imagenet_validation_set.getNumData() << " samples." << std::endl;
    }

    imagenet_validation_set.scale(scale);
    imagenet_validation_set.subtract_mean(subtract_mean);
    imagenet_validation_set.unit_variance(unit_variance);
    imagenet_validation_set.z_score(z_score);

    ///////////////////////////////////////////////////////////////////
    // load testing data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    imagenet_reader imagenet_testset(trainParams.MBSize);
    imagenet_testset.set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TestDir);
    imagenet_testset.set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile);
    imagenet_testset.set_use_percent(trainParams.PercentageTestingSamples);
    imagenet_testset.load();

    if (comm->am_world_master()) {
      std::cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << imagenet_testset.getNumData() << " samples." << std::endl;
    }
    imagenet_testset.scale(scale);
    imagenet_testset.subtract_mean(subtract_mean);
    imagenet_testset.unit_variance(unit_variance);
    imagenet_testset.z_score(z_score);

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
    dnn = new deep_neural_network(
      trainParams.MBSize,
      comm,
      new objective_functions::categorical_cross_entropy(comm),
      optimizer_fac);
    std::map<execution_mode, generic_data_reader *> data_readers = {
      std::make_pair(execution_mode::training,&imagenet_trainset),
      std::make_pair(execution_mode::validation, &imagenet_validation_set),
      std::make_pair(execution_mode::testing, &imagenet_testset)
    };
    dnn->add_metric(new metrics::categorical_accuracy(data_layout::DATA_PARALLEL, comm));
#ifdef PARTITIONED
    Layer *input_layer =
      new input_layer_partitioned_minibatch_parallel_io<>(
        comm,
        trainParams.MBSize,
        parallel_io,
        data_readers);
    dnn->add(input_layer);
#else
    Layer *input_layer =
      new input_layer_distributed_minibatch_parallel_io<data_layout::DATA_PARALLEL>(
        comm,
        trainParams.MBSize,
        parallel_io,
        data_readers);
    dnn->add(input_layer);
#endif
    // Layer 1 (convolutional)
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

    // Layer 4 (convolutional)
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

    // Layer 7 (convolutional)
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

    // Layer 8 (convolutional)
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
    }

    // Layer 9 (convolutional)
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
    {
      fully_connected_layer<data_layout::MODEL_PARALLEL> *fc =
        new fully_connected_layer<data_layout::MODEL_PARALLEL>(
          16,
          comm,
          trainParams.MBSize,
          4096,
          weight_initialization::he_normal,
          dnn->create_optimizer());
      fc->set_l2_regularization_factor(0.0005);
      dnn->add(fc);
      Layer *relu
        = new relu_layer<data_layout::MODEL_PARALLEL>(
          17,
          comm,
          trainParams.MBSize);
      dnn->add(relu);
      Layer *dropout_layer
        = new dropout<data_layout::MODEL_PARALLEL>(
          18,
          comm,
          trainParams.MBSize,
          0.5);
      dnn->add(dropout_layer);
    }

    // Layer 12 (fully-connected)
    {
      fully_connected_layer<data_layout::MODEL_PARALLEL> *fc =
        new fully_connected_layer<data_layout::MODEL_PARALLEL>(
          19,
          comm,
          trainParams.MBSize,
          4096,
          weight_initialization::he_normal,
          dnn->create_optimizer());
      fc->set_l2_regularization_factor(0.0005);
      dnn->add(fc);
      Layer *relu
        = new relu_layer<data_layout::MODEL_PARALLEL>(
          20,
          comm,
          trainParams.MBSize);
      dnn->add(relu);
      Layer *dropout_layer
        = new dropout<data_layout::MODEL_PARALLEL>(
          21,
          comm,
          trainParams.MBSize,
          0.5);
      dnn->add(dropout_layer);
    }

    // Layer 13 (softmax)
    {
      // Fully-connected without bias before softmax.
      fully_connected_layer<data_layout::MODEL_PARALLEL> *fc =
        new fully_connected_layer<data_layout::MODEL_PARALLEL>(
          22,
          comm,
          trainParams.MBSize,
          1000,
          weight_initialization::he_normal,
          dnn->create_optimizer(),
          false);
      fc->set_l2_regularization_factor(0.0005);
      dnn->add(fc);
      Layer *softmax =
        new softmax_layer<data_layout::MODEL_PARALLEL>(
          23,
          trainParams.MBSize,
          comm,
          dnn->create_optimizer());
      dnn->add(softmax);
    }

#ifdef PARTITIONED
    Layer *target_layer =
      new target_layer_partitioned_minibatch_parallel_io<>(
        comm,
        trainParams.MBSize,
        parallel_io,
        data_readers,
        true);
    dnn->add(target_layer);
#else
    Layer *target_layer =
      new target_layer_distributed_minibatch_parallel_io<data_layout::MODEL_PARALLEL>(
        comm,
        trainParams.MBSize,
        parallel_io,
        data_readers,
        true);
    dnn->add(target_layer);
#endif
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

    lbann_callback_imcomm imcomm_cb
      = lbann_callback_imcomm(static_cast<lbann_callback_imcomm::comm_type>
                              (trainParams.IntermodelCommMethod),
                              {1, 5, 9, 11, 13, 16, 19, 22}, &summarizer);
    dnn->add_callback(&imcomm_cb);

    lbann_callback_profiler profiler_cb;
    dnn->add_callback(&profiler_cb);

    dnn->setup();

    if (comm->am_world_master()) {
      std::cout << "Layer initialized:" << std::endl;
      for (uint n = 0; n < dnn->get_layers().size(); n++) {
        std::cout << "\tLayer[" << n << "]: " << dnn->get_layers()[n]->get_num_neurons() << std::endl;
      }
      std::cout << std::endl;
    }

    if (comm->am_world_master()) {
      std::cout << "Parameter settings:" << std::endl;
      std::cout << "\tBlock size: " << perfParams.BlockSize << std::endl;
      std::cout << "\tEpochs: " << trainParams.EpochCount << std::endl;
      std::cout << "\tMini-batch size: " << trainParams.MBSize << std::endl;
      std::cout << "\tLearning rate: " << trainParams.LearnRate << std::endl;
      std::cout << "\tEpoch count: " << trainParams.EpochCount << std::endl << std::endl;
      if(perfParams.MaxParIOSize == 0) {
        std::cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << std::endl;
      } else {
        std::cout << "\tMax Parallel I/O Fetch: " << perfParams.MaxParIOSize << std::endl;
      }
      std::cout << "\tDataset: " << trainParams.DatasetRootDir << std::endl;
    }

    // load parameters from file if available
    if (trainParams.LoadModel && trainParams.ParameterDir.length() > 0) {
      dnn->load_from_file(trainParams.ParameterDir);
    }

    comm->global_barrier();

    //************************************************************************
    // mainloop for train/validate
    //************************************************************************
    for (int epoch = 0; epoch < trainParams.EpochCount; epoch++) {
      dnn->train(1, true);
      dnn->evaluate(execution_mode::testing);
    }

    delete dnn;
  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);  /// Elemental exceptions
  }

  finalize(comm);

  return 0;
}
