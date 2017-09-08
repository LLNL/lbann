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
    bool use_gpus = Input("--use-gpus", "whether to use GPUs", true);
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

    if (comm->am_world_master()) {
      cout << "scale/subtract_mean/unit_variance/z_score: " << scale<<" "<<subtract_mean<<" "<<unit_variance<<" "<<z_score<<endl;
    }  

    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data (ImageNet)
    ///////////////////////////////////////////////////////////////////
    imagenet_reader imagenet_validation_set(imagenet_trainset); // Clone the training set object
    imagenet_validation_set.use_unused_index_set();

    if (comm->am_world_master()) {
      size_t num_train = imagenet_trainset.get_num_data();
      size_t num_validate = imagenet_validation_set.get_num_data();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      std::cout << "Training using " << train_percent << "% of the training data set, which is " << imagenet_trainset.get_num_data() << " samples." << std::endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << imagenet_validation_set.get_num_data() << " samples." << std::endl;
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
      std::cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << imagenet_testset.get_num_data() << " samples." << std::endl;
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
    std::unordered_set<uint> learning_layers;
    const DataType l2_regularization_factor = 1e-4;

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
    cudnn::cudnn_manager *cudnn = use_gpus ? new cudnn::cudnn_manager(comm, num_gpus) : NULL;
#else // __LIB_CUDNN
    cudnn::cudnn_manager *cudnn = NULL;
#endif // __LIB_CUDNN

    deep_neural_network *dnn = NULL;
    dnn = new deep_neural_network(
      trainParams.MBSize,
      comm,
      new objective_functions::cross_entropy(),
      optimizer_fac);
    std::map<execution_mode, generic_data_reader *> data_readers = {
      std::make_pair(execution_mode::training,&imagenet_trainset),
      std::make_pair(execution_mode::validation, &imagenet_validation_set),
      std::make_pair(execution_mode::testing, &imagenet_testset)
    };
    dnn->add_metric(new metrics::categorical_accuracy<data_layout::DATA_PARALLEL>(comm));
    dnn->add_metric(new metrics::top_k_categorical_accuracy<data_layout::DATA_PARALLEL>(5, comm));
#ifdef PARTITIONED
    Layer *input_layer =
      new input_layer_partitioned_minibatch<>(
        comm,
        parallel_io,
        data_readers);
    dnn->add(input_layer);
#else
    Layer *input_layer =
      new input_layer_distributed_minibatch<data_layout::DATA_PARALLEL>(
        comm,
        parallel_io,
        data_readers);
    dnn->add(input_layer);
#endif
    index++;

    // res1 module
    {

      convolution_layer<> *conv1
        = new convolution_layer<>(
          index,
          comm,
          2,
          64,
          7,
          3,
          2,
          weight_initialization::he_normal,
          optimizer_fac->create_optimizer(),
          false,
          cudnn);
      conv1->set_l2_regularization_factor(l2_regularization_factor);
      dnn->add(conv1);
      learning_layers.insert(index);
      index++;

      batch_normalization<data_layout::DATA_PARALLEL> *bn_conv1
        = new batch_normalization<data_layout::DATA_PARALLEL>(
          index,
          comm,
          0.9,
          1.0,
          0.0,
          1e-5,
          cudnn);
      dnn->add(bn_conv1);
      index++;

      relu_layer<data_layout::DATA_PARALLEL> *conv1_relu
        = new relu_layer<data_layout::DATA_PARALLEL>(index, comm, cudnn);
      dnn->add(conv1_relu);
      index++;

      pooling_layer<> *pool1
        = new pooling_layer<>(index, comm, 2, 3, 0, 2, pool_mode::max, cudnn);
      dnn->add(pool1);
      index++;

    }

    // Resnet module specificiations
    std::vector<int> intra_module_channels = {64, 128, 256, 512};
    std::vector<int> output_channels = {256, 512, 1024, 2048};
    std::vector<int> num_submodules = {3, 4, 6, 3};

    // Initialize resnet modules
    for(uint module = 0; module < output_channels.size(); ++module){

      // First submodule
      // Note: reduces spatial dimension (except in first module)
      {
        
        convolution_layer<> *conv1
          = new convolution_layer<>(
            index,
            comm,
            2,
            output_channels[module] + intra_module_channels[module],
            1,
            0,
            module == 0 ? 1 : 2,
            weight_initialization::he_normal,
            optimizer_fac->create_optimizer(),
            false,
            cudnn);
        conv1->set_l2_regularization_factor(l2_regularization_factor);
        dnn->add(conv1);
        learning_layers.insert(index);
        index++;

        batch_normalization<data_layout::DATA_PARALLEL> *bn1
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index,
            comm,
            0.9,
            1.0,
            0.0,
            1e-5,
            cudnn);
        dnn->add(bn1);
        index++;

        slice_layer<> *slice = new slice_layer<>(index, comm, {}, 0, {}, cudnn);
        dnn->add(slice);
        index++;

        relu_layer<data_layout::DATA_PARALLEL> *relu1
          = new relu_layer<data_layout::DATA_PARALLEL>(index, comm, cudnn);
        dnn->add(relu1);
        index++;

        convolution_layer<> *conv2
          = new convolution_layer<>(
            index,
            comm,
            2,
            intra_module_channels[module],
            3,
            1,
            1,
            weight_initialization::he_normal,
            optimizer_fac->create_optimizer(),
            false,
            cudnn);
        conv2->set_l2_regularization_factor(l2_regularization_factor);
        dnn->add(conv2);
        learning_layers.insert(index);
        index++;

        batch_normalization<data_layout::DATA_PARALLEL> *bn2
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index,
            comm,
            0.9,
            1.0,
            0.0,
            1e-5,
            cudnn);
        dnn->add(bn2);
        index++;

        relu_layer<data_layout::DATA_PARALLEL> *relu2
          = new relu_layer<data_layout::DATA_PARALLEL>(index, comm, cudnn);
        dnn->add(relu2);
        index++;

        convolution_layer<> *conv3
          = new convolution_layer<>(
            index,
            comm,
            2,
            output_channels[module],
            1,
            0,
            1,
            weight_initialization::he_normal,
            optimizer_fac->create_optimizer(),
            false,
            cudnn);
        conv3->set_l2_regularization_factor(l2_regularization_factor);
        dnn->add(conv3);
        learning_layers.insert(index);
        index++;

        batch_normalization<data_layout::DATA_PARALLEL> *bn3
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index,
            comm,
            0.9,
            1.0,
            0.0,
            1e-5,
            cudnn);
        dnn->add(bn3);
        index++;

        sum_layer<> *sum = new sum_layer<>(index, comm, {}, cudnn);
        dnn->add(sum);
        index++;

        relu_layer<data_layout::DATA_PARALLEL> *relu3
          = new relu_layer<data_layout::DATA_PARALLEL>(index, comm, cudnn);
        dnn->add(relu3);
        index++;

        slice->push_back_child(sum, 0);
        slice->push_back_child(relu1, output_channels[module]);
        sum->add_parent(slice);
        // sum->add_parent(bn3);

      }

      // Additional submodules
      for(int submodule = 1; submodule < num_submodules[module]; ++submodule){
        
        split_layer<> *split = new split_layer<>(index, comm, {}, cudnn);
        dnn->add(split);
        index++;

        convolution_layer<> *conv1
          = new convolution_layer<>(
            index,
            comm,
            2,
            intra_module_channels[module],
            1,
            0,
            1,
            weight_initialization::he_normal,
            optimizer_fac->create_optimizer(),
            false,
            cudnn);
        conv1->set_l2_regularization_factor(l2_regularization_factor);
        dnn->add(conv1);
        learning_layers.insert(index);
        index++;

        batch_normalization<data_layout::DATA_PARALLEL> *bn1
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index,
            comm,
            0.9,
            1.0,
            0.0,
            1e-5,
            cudnn);
        dnn->add(bn1);
        index++;

        relu_layer<data_layout::DATA_PARALLEL> *relu1
          = new relu_layer<data_layout::DATA_PARALLEL>(index, comm, cudnn);
        dnn->add(relu1);
        index++;

        convolution_layer<> *conv2
          = new convolution_layer<>(
            index,
            comm,
            2,
            intra_module_channels[module],
            3,
            1,
            1,
            weight_initialization::he_normal,
            optimizer_fac->create_optimizer(),
            false,
            cudnn);
        conv2->set_l2_regularization_factor(l2_regularization_factor);
        dnn->add(conv2);
        learning_layers.insert(index);
        index++;

        batch_normalization<data_layout::DATA_PARALLEL> *bn2
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index,
            comm,
            0.9,
            1.0,
            0.0,
            1e-5,
            cudnn);
        dnn->add(bn2);
        index++;

        relu_layer<data_layout::DATA_PARALLEL> *relu2
          = new relu_layer<data_layout::DATA_PARALLEL>(index, comm, cudnn);
        dnn->add(relu2);
        index++;

        convolution_layer<> *conv3
          = new convolution_layer<>(
            index,
            comm,
            2,
            output_channels[module],
            1,
            0,
            1,
            weight_initialization::he_normal,
            optimizer_fac->create_optimizer(),
            false,
            cudnn);
        conv3->set_l2_regularization_factor(l2_regularization_factor);
        dnn->add(conv3);
        learning_layers.insert(index);
        index++;

        batch_normalization<data_layout::DATA_PARALLEL> *bn3
          = new batch_normalization<data_layout::DATA_PARALLEL>(
            index,
            comm,
            0.9,
            1.0,
            0.0,
            1e-5,
            cudnn);
        dnn->add(bn3);
        index++;

        sum_layer<> *sum = new sum_layer<>(index, comm, {}, cudnn);
        dnn->add(sum);
        index++;

        relu_layer<data_layout::DATA_PARALLEL> *relu3
          = new relu_layer<data_layout::DATA_PARALLEL>(index, comm, cudnn);
        dnn->add(relu3);
        index++;

        split->add_child(sum);
        // split->add_child(conv1);
        sum->add_parent(split);
        // sum->add_parent(bn3);

      }

    }

    pooling_layer<> *pool5
      = new pooling_layer<>(index, comm, 2, 8, 0, 1, pool_mode::average, cudnn);
    dnn->add(pool5);
    index++;

    fully_connected_layer<data_layout::MODEL_PARALLEL> *fc1000
      = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
        index,
        comm,
        1000,
        weight_initialization::he_normal,
        dnn->create_optimizer(),
        false);
    fc1000->set_l2_regularization_factor(l2_regularization_factor);
    dnn->add(fc1000);
    learning_layers.insert(index);
    index++;

    softmax_layer<data_layout::MODEL_PARALLEL> *softmax
      = new softmax_layer<data_layout::MODEL_PARALLEL>(
        index,
        comm);
      dnn->add(softmax);
      index++;

#ifdef PARTITIONED
    Layer *target_layer =
      new target_layer_partitioned_minibatch<>(
        comm,
        parallel_io,
        data_readers,
        true);
    dnn->add(target_layer);
    index++;
#else
    Layer *target_layer =
      new target_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(
        comm,
        parallel_io,
        data_readers,
        true);
    dnn->add(target_layer);
    index++;
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
                              learning_layers,
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

    if (comm->am_world_master()) {
      optimizer *o = optimizer_fac->create_optimizer();
      cout << "\nOptimizer:\n" << o->get_description() << endl << endl;
      std::vector<Layer *>& layers = dnn->get_layers();
      for (size_t h=0; h<layers.size(); h++) {
        std::cout << h << " " << layers[h]->get_description() << endl;
      }
    }

    //************************************************************************
    // train and validate
    //************************************************************************
    dnn->train(trainParams.EpochCount);
    dnn->evaluate(execution_mode::testing);

    delete dnn;
  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);  /// Elemental exceptions
  }

  finalize(comm);

  return 0;
}
