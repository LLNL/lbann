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
// resnet50.cpp - ResNet-50 application for ImageNet classification
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/data_readers/image_utils.hpp"

using namespace lbann;

// train/test data info
const string g_ImageNet_TrainDir = "resized_256x256/train/";
const string g_ImageNet_ValDir = "resized_256x256/val/";
const string g_ImageNet_TestDir = "resized_256x256/val/";
const string g_ImageNet_LabelDir = "labels/";
const int g_ImageNet_Width = 256;
const int g_ImageNet_Height = 256;

#define PARTITIONED

int main(int argc, char *argv[]) {
  lbann_comm *comm = initialize(argc, argv, 42);

#ifdef EL_USE_CUBLAS
  El::GemmUseGPU(32,32,32);
#endif

  try {
    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/ILSVRC2012/";
    trainParams.LearnRate = 1e-2;
    trainParams.DropOut = 0.5;
    trainParams.ProcsPerModel = 1;
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

    // Initialize parameters
    int index = 0;
    const int num_dims = 2;

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
    dnn->add_metric(new metrics::categorical_accuracy<data_layout::DATA_PARALLEL>(comm));
#ifdef PARTITIONED
    Layer *input_layer =
      new input_layer_partitioned_minibatch<>(
        comm,
        trainParams.MBSize,
        parallel_io,
        data_readers);
    dnn->add(input_layer);
#else
    Layer *input_layer =
      new input_layer_distributed_minibatch<data_layout::DATA_PARALLEL>(
        comm,
        trainParams.MBSize,
        parallel_io,
        data_readers);
    dnn->add(input_layer);
#endif
    index++;

    // res1 module
    {

      optimizer *conv_optimizer = optimizer_fac->create_optimizer();
      const int output_channels = 64;
      const int filter_dims[] = {7, 7};
      const int conv_pads[] = {3, 3};
      const int conv_strides[] = {2, 2};
      convolution_layer<> *conv1
        = new convolution_layer<>(
          index++,
          comm,
          trainParams.MBSize,
          num_dims,
          output_channels,
          filter_dims,
          conv_pads,
          conv_strides,
          weight_initialization::he_normal,
          conv_optimizer,
          false,
          cudnn);
      conv1->set_l2_regularization_factor(1e-4);
      dnn->add(conv1);

      batch_normalization<data_layout::DATA_PARALLEL> *bn_conv1
        = new batch_normalization<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize);
      dnn->add(bn_conv1);

      relu_layer<data_layout::DATA_PARALLEL> *conv1_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(conv1_relu);

      const int pool_dims[] = {3, 3};
      const int pool_pads[] = {0, 0};
      const int pool_strides[] = {2, 2};
      pooling_layer<> *pool1
        = new pooling_layer<>(
          index++,
          comm,
          trainParams.MBSize,
          num_dims,
          pool_dims,
          pool_pads,
          pool_strides,
          pool_mode::max,
          cudnn);
      dnn->add(pool1);

    }

    // res2a module
    {

      split_layer<data_layout::DATA_PARALLEL> *res2a_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res2a_split);

      // res2a_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 64;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res2a_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res2a_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res2a_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res2a_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res2a_branch2a_relu);
        
      }

      // res2a_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 64;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res2a_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res2a_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res2a_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res2a_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res2a_branch2b_relu);
        
      }

      concatenation_layer<data_layout::DATA_PARALLEL> *res2a_concat
        = new concatenation_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          0,
          NULL);
      dnn->add(res2a_concat);
      res2a_split->add_child(res2a_concat);
      res2a_concat->push_back_parent(res2a_split);

      // res2a_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res2a_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res2a_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res2a_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);

        relu_layer<data_layout::DATA_PARALLEL> *res2a_branch2c_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res2a_branch2c_relu);
        
      }      

    }

    // res2b module
    {

      split_layer<data_layout::DATA_PARALLEL> *res2b_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res2b_split);

      // res2b_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 64;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res2b_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res2b_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res2b_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res2b_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res2b_branch2a_relu);
        
      }

      // res2b_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 64;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res2b_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res2b_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res2b_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res2b_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res2b_branch2b_relu);
        
      }

      // res2b_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res2b_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res2b_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res2b_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res2b_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res2b_sum);
      res2b_split->add_child(res2b_sum);
      res2b_sum->add_parent(res2b_split);

      relu_layer<data_layout::DATA_PARALLEL> *res2b_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res2b_relu);

    }

    // res2c module
    {

      split_layer<data_layout::DATA_PARALLEL> *res2c_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res2c_split);

      // res2c_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 64;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res2c_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res2c_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res2c_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res2c_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res2c_branch2a_relu);
        
      }

      // res2c_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 64;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res2c_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res2c_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res2c_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res2c_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res2c_branch2b_relu);
        
      }

      // res2c_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res2c_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res2c_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res2c_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res2c_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res2c_sum);
      res2c_split->add_child(res2c_sum);
      res2c_sum->add_parent(res2c_split);

      relu_layer<data_layout::DATA_PARALLEL> *res2c_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res2c_relu);

    }

    // res3a module
    {

      split_layer<data_layout::DATA_PARALLEL> *res3a_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res3a_split);

      // res3a_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 128;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3a_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3a_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res3a_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res3a_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res3a_branch2a_relu);
        
      }

      // res3a_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 128;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3a_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3a_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res3a_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res3a_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res3a_branch2b_relu);
        
      }

      concatenation_layer<data_layout::DATA_PARALLEL> *res3a_concat
        = new concatenation_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          0,
          NULL);
      dnn->add(res3a_concat);
      res3a_split->add_child(res3a_concat);
      res3a_concat->push_back_parent(res3a_split);

      // res3a_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 512;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {2, 2};
        convolution_layer<> *res3a_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3a_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res3a_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);

        relu_layer<data_layout::DATA_PARALLEL> *res3a_branch2c_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res3a_branch2c_relu);
        
      }      

    }    

    // res3b module
    {

      split_layer<data_layout::DATA_PARALLEL> *res3b_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res3b_split);

      // res3b_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 128;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3b_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3b_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res3b_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res3b_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res3b_branch2a_relu);
        
      }

      // res3b_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 128;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3b_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3b_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res3b_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res3b_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res3b_branch2b_relu);
        
      }

      // res3b_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 512;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3b_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3b_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res3b_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res3b_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res3b_sum);
      res3b_split->add_child(res3b_sum);
      res3b_sum->add_parent(res3b_split);

      relu_layer<data_layout::DATA_PARALLEL> *res3b_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res3b_relu);

    }

    // res3c module
    {

      split_layer<data_layout::DATA_PARALLEL> *res3c_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res3c_split);

      // res3c_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 128;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3c_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3c_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res3c_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res3c_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res3c_branch2a_relu);
        
      }

      // res3c_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 128;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3c_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3c_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res3c_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res3c_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res3c_branch2b_relu);
        
      }

      // res3c_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 512;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3c_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3c_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res3c_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res3c_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res3c_sum);
      res3c_split->add_child(res3c_sum);
      res3c_sum->add_parent(res3c_split);

      relu_layer<data_layout::DATA_PARALLEL> *res3c_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res3c_relu);

    }    

    // res3d module
    {

      split_layer<data_layout::DATA_PARALLEL> *res3d_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res3d_split);

      // res3d_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 128;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3d_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3d_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res3d_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res3d_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res3d_branch2a_relu);
        
      }

      // res3d_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 128;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3d_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3d_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res3d_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res3d_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res3d_branch2b_relu);
        
      }

      // res3d_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 512;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res3d_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res3d_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res3d_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res3d_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res3d_sum);
      res3d_split->add_child(res3d_sum);
      res3d_sum->add_parent(res3d_split);

      relu_layer<data_layout::DATA_PARALLEL> *res3d_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res3d_relu);

    }

    // res4a module
    {

      split_layer<data_layout::DATA_PARALLEL> *res4a_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4a_split);

      // res4a_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4a_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4a_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res4a_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res4a_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4a_branch2a_relu);
        
      }

      // res4a_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4a_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4a_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res4a_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res4a_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4a_branch2b_relu);
        
      }

      concatenation_layer<data_layout::DATA_PARALLEL> *res4a_concat
        = new concatenation_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          0,
          NULL);
      dnn->add(res4a_concat);
      res4a_split->add_child(res4a_concat);
      res4a_concat->push_back_parent(res4a_split);

      // res4a_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 1024;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {2, 2};
        convolution_layer<> *res4a_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4a_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res4a_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);

        relu_layer<data_layout::DATA_PARALLEL> *res4a_branch2c_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4a_branch2c_relu);
        
      }      

    }

    // res4b module
    {

      split_layer<data_layout::DATA_PARALLEL> *res4b_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4b_split);

      // res4b_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4b_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4b_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res4b_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res4b_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4b_branch2a_relu);
        
      }

      // res4b_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4b_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4b_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res4b_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res4b_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4b_branch2b_relu);
        
      }

      // res4b_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 1024;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4b_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4b_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res4b_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res4b_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4b_sum);
      res4b_split->add_child(res4b_sum);
      res4b_sum->add_parent(res4b_split);

      relu_layer<data_layout::DATA_PARALLEL> *res4b_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res4b_relu);

    }

    // res4c module
    {

      split_layer<data_layout::DATA_PARALLEL> *res4c_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4c_split);

      // res4c_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4c_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4c_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res4c_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res4c_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4c_branch2a_relu);
        
      }

      // res4c_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4c_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4c_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res4c_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res4c_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4c_branch2b_relu);
        
      }

      // res4c_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 1024;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4c_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4c_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res4c_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res4c_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4c_sum);
      res4c_split->add_child(res4c_sum);
      res4c_sum->add_parent(res4c_split);

      relu_layer<data_layout::DATA_PARALLEL> *res4c_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res4c_relu);

    }

    // res4d module
    {

      split_layer<data_layout::DATA_PARALLEL> *res4d_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4d_split);

      // res4d_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4d_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4d_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res4d_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res4d_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4d_branch2a_relu);
        
      }

      // res4d_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4d_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4d_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res4d_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res4d_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4d_branch2b_relu);
        
      }

      // res4d_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 1024;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4d_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4d_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res4d_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res4d_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4d_sum);
      res4d_split->add_child(res4d_sum);
      res4d_sum->add_parent(res4d_split);

      relu_layer<data_layout::DATA_PARALLEL> *res4d_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res4d_relu);

    }

    // res4e module
    {

      split_layer<data_layout::DATA_PARALLEL> *res4e_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4e_split);

      // res4e_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4e_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4e_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res4e_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res4e_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4e_branch2a_relu);
        
      }

      // res4e_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4e_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4e_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res4e_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res4e_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4e_branch2b_relu);
        
      }

      // res4e_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 1024;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4e_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4e_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res4e_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res4e_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4e_sum);
      res4e_split->add_child(res4e_sum);
      res4e_sum->add_parent(res4e_split);

      relu_layer<data_layout::DATA_PARALLEL> *res4e_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res4e_relu);

    }

    // res4f module
    {

      split_layer<data_layout::DATA_PARALLEL> *res4f_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4f_split);

      // res4f_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4f_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4f_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res4f_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res4f_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4f_branch2a_relu);
        
      }

      // res4f_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 256;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4f_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4f_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res4f_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res4f_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res4f_branch2b_relu);
        
      }

      // res4f_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 1024;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res4f_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res4f_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res4f_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res4f_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res4f_sum);
      res4f_split->add_child(res4f_sum);
      res4f_sum->add_parent(res4f_split);

      relu_layer<data_layout::DATA_PARALLEL> *res4f_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res4f_relu);

    }

    // res5a module
    {

      split_layer<data_layout::DATA_PARALLEL> *res5a_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res5a_split);

      // res5a_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 512;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res5a_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res5a_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res5a_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res5a_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res5a_branch2a_relu);
        
      }

      // res5a_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 512;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res5a_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res5a_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res5a_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res5a_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res5a_branch2b_relu);
        
      }

      concatenation_layer<data_layout::DATA_PARALLEL> *res5a_concat
        = new concatenation_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          0,
          NULL);
      dnn->add(res5a_concat);
      res5a_split->add_child(res5a_concat);
      res5a_concat->push_back_parent(res5a_split);

      // res5a_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 2048;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {2, 2};
        convolution_layer<> *res5a_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res5a_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res5a_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);

        relu_layer<data_layout::DATA_PARALLEL> *res5a_branch2c_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res5a_branch2c_relu);
        
      }      

    }
    
    // res5b module
    {

      split_layer<data_layout::DATA_PARALLEL> *res5b_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res5b_split);

      // res5b_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 512;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res5b_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res5b_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res5b_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res5b_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res5b_branch2a_relu);
        
      }

      // res5b_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 512;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res5b_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res5b_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res5b_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res5b_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res5b_branch2b_relu);
        
      }

      // res5b_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 2048;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res5b_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res5b_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res5b_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res5b_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res5b_sum);
      res5b_split->add_child(res5b_sum);
      res5b_sum->add_parent(res5b_split);

      relu_layer<data_layout::DATA_PARALLEL> *res5b_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res5b_relu);

    }

    // res5c module
    {

      split_layer<data_layout::DATA_PARALLEL> *res5c_split
        = new split_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res5c_split);

      // res5c_branch2a module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 512;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res5c_branch2a
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res5c_branch2a->set_l2_regularization_factor(1e-4);
        dnn->add(res5c_branch2a);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2a
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2a);

        relu_layer<data_layout::DATA_PARALLEL> *res5c_branch2a_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res5c_branch2a_relu);
        
      }

      // res5c_branch2b module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 512;
        const int filter_dims[] = {3, 3};
        const int conv_pads[] = {1, 1};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res5c_branch2b
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res5c_branch2b->set_l2_regularization_factor(1e-4);
        dnn->add(res5c_branch2b);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2b
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2b);

        relu_layer<data_layout::DATA_PARALLEL> *res5c_branch2b_relu
          = new relu_layer<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize,
            cudnn);
        dnn->add(res5c_branch2b_relu);
        
      }

      // res5c_branch2c module
      {

        optimizer *conv_optimizer = optimizer_fac->create_optimizer();
        const int output_channels = 2048;
        const int filter_dims[] = {1, 1};
        const int conv_pads[] = {0, 0};
        const int conv_strides[] = {1, 1};
        convolution_layer<> *res5c_branch2c
          = new convolution_layer<>(
            index++,
            comm,
            trainParams.MBSize,
            num_dims,
            output_channels,
            filter_dims,
            conv_pads,
            conv_strides,
            weight_initialization::he_normal,
            conv_optimizer,
            false,
            cudnn);
        res5c_branch2c->set_l2_regularization_factor(1e-4);
        dnn->add(res5c_branch2c);

        batch_normalization<data_layout::DATA_PARALLEL> *bn2a_branch2c
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index++,
            comm,
            trainParams.MBSize);
        dnn->add(bn2a_branch2c);
        
      }      

      sum_layer<data_layout::DATA_PARALLEL> *res5c_sum
        = new sum_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          {},
          NULL);
      dnn->add(res5c_sum);
      res5c_split->add_child(res5c_sum);
      res5c_sum->add_parent(res5c_split);

      relu_layer<data_layout::DATA_PARALLEL> *res5c_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(
          index++,
          comm,
          trainParams.MBSize,
          cudnn);
      dnn->add(res5c_relu);

    }

    const int pool_dims[] = {8, 8};
    const int pool_pads[] = {0, 0};
    const int pool_strides[] = {1, 1};
    pooling_layer<> *pool5
      = new pooling_layer<>(
        index++,
        comm,
        trainParams.MBSize,
        num_dims,
        pool_dims,
        pool_pads,
        pool_strides,
        pool_mode::average,
        cudnn);
    dnn->add(pool5);

    fully_connected_layer<data_layout::MODEL_PARALLEL> *fc1000
      = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
        index++,
        comm,
        trainParams.MBSize,
        1000,
        weight_initialization::he_normal,
        dnn->create_optimizer(),
        false);
    fc1000->set_l2_regularization_factor(1e-4);
    dnn->add(fc1000);

    softmax_layer<data_layout::MODEL_PARALLEL> *softmax
      = new softmax_layer<data_layout::MODEL_PARALLEL>(
        index++,
        comm,
        trainParams.MBSize,
        dnn->create_optimizer());
      dnn->add(softmax);

#ifdef PARTITIONED
    Layer *target_layer =
      new target_layer_partitioned_minibatch<>(
        comm,
        trainParams.MBSize,
        parallel_io,
        data_readers,
        true);
    dnn->add(target_layer);
#else
    Layer *target_layer =
      new target_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(
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
    lbann_callback_summary summary_cb(&summarizer);
    dnn->add_callback(&summary_cb);
    // lbann_callback_io io_cb({0});
    // dnn->add_callback(&io_cb);

    lbann_callback_imcomm imcomm_cb
      = lbann_callback_imcomm(static_cast<lbann_callback_imcomm::comm_type>
                              (trainParams.IntermodelCommMethod),
                              {1, 6, 9, 13, 17, 20, 23, 28, 31, 34, 39, 42, 46, 50, 53, 56, 61, 64, 67, 72, 75, 78, 83, 86, 90, 94, 97, 100, 105, 108, 111, 116, 119, 122, 127, 130, 133, 138, 141, 144, 149, 152, 156, 160, 163, 166, 171, 174, 177, 182},
                              &summarizer);
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
