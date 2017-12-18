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
// dnn_mnist.cpp - DNN application for mnist
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_mnist.hpp"
#include "lbann/callbacks/callback_dump_weights.hpp"
#include "lbann/callbacks/callback_dump_activations.hpp"
#include "lbann/callbacks/callback_dump_gradients.hpp"
#include "lbann/lbann.hpp"

using namespace lbann;
using namespace std;
using namespace El;

/// Main function
int main(int argc, char *argv[]) {
  lbann_comm *comm = initialize(argc, argv, 42);

#ifdef EL_USE_CUBLAS
  El::GemmUseGPU(32,32,32);
#endif

  try {
    // Get data files
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

    //determine if we're going to scale, subtract mean, etc;
    //scaling/standardization is on a per-example basis (computed independantly
    //for each image)
    bool scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
    bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", false);
    bool unit_variance = Input("--unit-variance", "standardize to unit-variance", false);

    //if set to true, above three settings have no effect
    bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

#ifdef LBANN_HAS_CUDNN
    // Number of GPUs per node to use
    int num_gpus = Input("--num-gpus", "number of GPUs to use", -1);
#endif

    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////

    // Initialize parameter defaults
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/MNIST/";
    trainParams.EpochCount = 20;
    trainParams.MBSize = 128;
    trainParams.LearnRate = 0.01;
    trainParams.DropOut = -1.0f;
    trainParams.ProcsPerModel = 0;
    trainParams.PercentageTrainingSamples = 1.0;
    trainParams.PercentageValidationSamples = 0.1;
    PerformanceParams perfParams;
    perfParams.BlockSize = 256;

    // Parse command-line inputs
    trainParams.parse_params();
    perfParams.parse_params();

    ProcessInput();
    PrintInputReport();

    // Set algorithmic blocksize
    SetBlocksize(perfParams.BlockSize);

    // Set up the communicator and get the grid.
    comm->split_models(trainParams.ProcsPerModel);
    Grid& grid = comm->get_model_grid();
    if (comm->am_world_master()) {
      std::cout << "Number of models: " << comm->get_num_models() << std::endl;
      std::cout << "Grid is " << grid.Height() << " x " << grid.Width() << std::endl;
      std::cout << std::endl;
    }

    int parallel_io = perfParams.MaxParIOSize;
    if (parallel_io == 0) {
      if (comm->am_world_master()) {
        std::cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() <<
             " (Limited to # Processes)" << std::endl;
      }
      parallel_io = comm->get_procs_per_model();
    } else {
      if (comm->am_world_master()) {
        std::cout << "\tMax Parallel I/O Fetch: " << parallel_io << std::endl;
      }
    }

    ///////////////////////////////////////////////////////////////////
    // load training data (MNIST)
    ///////////////////////////////////////////////////////////////////
    auto* mnist_trainset = new mnist_reader(true);
    mnist_trainset->set_file_dir(trainParams.DatasetRootDir);
    mnist_trainset->set_data_filename(g_MNIST_TrainImageFile);
    mnist_trainset->set_label_filename(g_MNIST_TrainLabelFile);
    mnist_trainset->set_validation_percent(trainParams.PercentageValidationSamples);
    mnist_trainset->load();

    mnist_trainset->scale(scale);
    mnist_trainset->subtract_mean(subtract_mean);
    mnist_trainset->unit_variance(unit_variance);
    mnist_trainset->z_score(z_score);

    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data (MNIST)
    ///////////////////////////////////////////////////////////////////
    auto* mnist_validation_set = new mnist_reader(*mnist_trainset); // Clone the training set object
    mnist_validation_set->use_unused_index_set();

    if (comm->am_world_master()) {
      size_t num_train = mnist_trainset->get_num_data();
      size_t num_validate = mnist_validation_set->get_num_data();
      double validate_percent = num_validate*100.0 / (num_train+num_validate);
      double train_percent = num_train*100.0 / (num_train+num_validate);
      std::cout << "Training using " << train_percent << "% of the training data set, which is " << mnist_trainset->get_num_data() << " samples." << std::endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << mnist_validation_set->get_num_data() << " samples." << std::endl;
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data (MNIST)
    ///////////////////////////////////////////////////////////////////
    auto* mnist_testset = new mnist_reader(true);
    mnist_testset->set_file_dir(trainParams.DatasetRootDir);
    mnist_testset->set_data_filename(g_MNIST_TestImageFile);
    mnist_testset->set_label_filename(g_MNIST_TestLabelFile);
    mnist_testset->set_use_percent(trainParams.PercentageTestingSamples);
    mnist_testset->load();

    mnist_testset->scale(scale);
    mnist_testset->subtract_mean(subtract_mean);
    mnist_testset->unit_variance(unit_variance);
    mnist_testset->z_score(z_score);

    if (comm->am_world_master()) {
      std::cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << mnist_testset->get_num_data() << " samples." << std::endl;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////

    // Initialize optimizer
    optimizer *default_optimizer;
    if (trainParams.LearnRateMethod == 1) { // Adagrad
      default_optimizer = new adagrad(comm, trainParams.LearnRate);
    } else if (trainParams.LearnRateMethod == 2) { // RMSprop
      default_optimizer = new rmsprop(comm, trainParams.LearnRate, 0.9);
    } else if (trainParams.LearnRateMethod == 3) { // Adam
      default_optimizer = new adam(comm, trainParams.LearnRate);
    } else {
      default_optimizer = new sgd(comm, trainParams.LearnRate, 0.9, true);
    }

    // Initialize network
#ifdef LBANN_HAS_CUDNN
    cudnn::cudnn_manager *cudnn = new cudnn::cudnn_manager(comm, num_gpus);
#else // LBANN_HAS_CUDNN
    cudnn::cudnn_manager *cudnn = nullptr;
#endif // LBANN_HAS_CUDNN
    auto *obj_fn = new objective_function();
    obj_fn->add_term(new cross_entropy());
    sequential_model dnn(comm, trainParams.MBSize, obj_fn, default_optimizer);
    dnn.add_metric(new categorical_accuracy_metric(comm));
    std::map<execution_mode, generic_data_reader *> data_readers = {std::make_pair(execution_mode::training, mnist_trainset),
                                                           std::make_pair(execution_mode::validation, mnist_validation_set),
                                                           std::make_pair(execution_mode::testing, mnist_testset)
                                                          };


    //first layer
    Layer *ilayer = new input_layer_distributed_minibatch<data_layout::DATA_PARALLEL>(comm, parallel_io, data_readers);
    dnn.add_layer(ilayer);

    // First convolution layer
    {
      int numDims = 2;
      int outputChannels = 32;
      std::vector<int> filterDims = {3, 3};
      std::vector<int> convPads = {0, 0};
      std::vector<int> convStrides = {1, 1};

      convolution_layer<> *layer
        = new convolution_layer<>(comm,
                                  numDims,
                                  outputChannels,
                                  filterDims,
                                  convPads,
                                  convStrides,
                                  true,
                                  cudnn);
      dnn.add_layer(layer);

      Layer *relu = new relu_layer<data_layout::DATA_PARALLEL>(comm,
                                                               cudnn);
      dnn.add_layer(relu);
    }

    // Second convolution layer
    {
      int numDims = 2;
      int outputChannels = 32;
      std::vector<int> filterDims = {3, 3};
      std::vector<int> convPads = {0, 0};
      std::vector<int> convStrides = {1, 1};

      convolution_layer<> *layer
        = new convolution_layer<>(comm,
                                  numDims,
                                  outputChannels,
                                  filterDims,
                                  convPads,
                                  convStrides,
                                  true,
                                  cudnn);
      dnn.add_layer(layer);

      Layer *relu = new relu_layer<data_layout::DATA_PARALLEL>(comm,
                                                               cudnn);
      dnn.add_layer(relu);
    }

    // Pooling layer
    {
      int numDims = 2;
      std::vector<int> poolWindowDims = {2, 2};
      std::vector<int> poolPads = {0, 0};
      std::vector<int> poolStrides = {2, 2};
      pool_mode poolMode = pool_mode::max;
      pooling_layer<> *layer
        = new pooling_layer<>(comm,
                              numDims,
                              poolWindowDims,
                              poolPads,
                              poolStrides,
                              poolMode,
                              cudnn);
      dnn.add_layer(layer);
    }

    // First fully connected layer
    {
      Layer *fc = new fully_connected_layer<data_layout::MODEL_PARALLEL>(comm,
                                                                         128);
      dnn.add_layer(fc);
      Layer *relu = new relu_layer<data_layout::MODEL_PARALLEL>(comm,
                                                                nullptr);
      dnn.add_layer(relu);
      Layer *dropout1 = new dropout<data_layout::MODEL_PARALLEL>(comm,
                                                                 0.5);
      dnn.add_layer(dropout1);
    }

    // Second fully connected layer
    {
      Layer *fc = new fully_connected_layer<data_layout::MODEL_PARALLEL>(comm,
                                                                         10,
                                                                         nullptr,
                                                                         false);
      dnn.add_layer(fc);
    }

    // Softmax layer
    Layer *sl = new softmax_layer<data_layout::MODEL_PARALLEL>(comm);
    dnn.add_layer(sl);

    // Target layer
    Layer *tlayer = new target_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(comm, dynamic_cast<input_layer*>(ilayer), parallel_io, data_readers, true);
    dnn.add_layer(tlayer);

    auto* print_cb = new lbann_callback_print;
    dnn.add_callback(print_cb);
    lbann_callback_dump_weights *dump_weights_cb = nullptr;
    lbann_callback_dump_activations *dump_activations_cb = nullptr;
    lbann_callback_dump_gradients *dump_gradients_cb = nullptr;
    if (trainParams.DumpWeights) {
      dump_weights_cb = new lbann_callback_dump_weights(
        trainParams.DumpDir);
      dnn.add_callback(dump_weights_cb);
    }
    if (trainParams.DumpActivations) {
      dump_activations_cb = new lbann_callback_dump_activations(
        trainParams.DumpDir);
      dnn.add_callback(dump_activations_cb);
    }
    if (trainParams.DumpGradients) {
      dump_gradients_cb = new lbann_callback_dump_gradients(
        trainParams.DumpDir);
      dnn.add_callback(dump_gradients_cb);
    }
    // lbann_callback_io* io_cb = new lbann_callback_io({0,3});
    // dnn.add_callback(io_cb);
    // lbann_callback_debug* debug_cb = new lbann_callback_debug(execution_mode::testing);
    // dnn.add_callback(debug_cb);

    if (comm->am_world_master()) {
      std::cout << "Parameter settings:" << std::endl;
      std::cout << "\tMini-batch size: " << trainParams.MBSize << std::endl;
      std::cout << "\tLearning rate: " << trainParams.LearnRate << std::endl << std::endl;
      std::cout << "\tEpoch count: " << trainParams.EpochCount << std::endl;
    }

    if (comm->am_world_master()) {
      cout << "\nOptimizer:\n" << default_optimizer->get_description() << endl << endl;
      const std::vector<Layer *>& layers = dnn.get_layers();
      for (size_t h=0; h<layers.size(); h++) {
        std::cout << h << " " << layers[h]->get_description() << endl;
      }
    }
    ///////////////////////////////////////////////////////////////////
    // training and testing
    ///////////////////////////////////////////////////////////////////

    // Initialize the model's data structures
    dnn.setup();

    // set checkpoint directory and checkpoint interval
    //dnn.set_checkpoint_dir(trainParams.ParameterDir);
    //dnn.set_checkpoint_epochs(trainParams.CkptEpochs);
    //dnn.set_checkpoint_steps(trainParams.CkptSteps);
    //dnn.set_checkpoint_secs(trainParams.CkptSecs);

    // restart model from checkpoint if we have one
    // dnn.restartShared();

    // train
    dnn.train(trainParams.EpochCount);
    
    // testing
    dnn.evaluate(execution_mode::testing);

    // Free dynamically allocated memory
    // delete tlayer;  // Causes segfault
    // delete ilayer;  // Causes segfault
    // delete lfac;  // Causes segfault
    if (trainParams.DumpWeights) {
      delete dump_weights_cb;
    }
    if (trainParams.DumpActivations) {
      delete dump_activations_cb;
    }
    if (trainParams.DumpGradients) {
      delete dump_gradients_cb;
    }
  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);  /// Elemental exceptions
  }

  finalize(comm);

  return 0;
}
