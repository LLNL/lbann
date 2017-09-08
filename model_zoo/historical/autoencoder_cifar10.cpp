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
// dnn_cifar10.cpp - DNN application for cifar10
////////////////////////////////////////////////////////////////////////////////

//#include "lbann/data_readers/data_reader_cifar10.hpp"
#include "lbann/callbacks/callback_dump_weights.hpp"
#include "lbann/callbacks/callback_dump_activations.hpp"
#include "lbann/callbacks/callback_dump_gradients.hpp"
#include "lbann/lbann.hpp"

using namespace lbann;

const string g_cifar10_dir = "/p/lscratchf/brainusr/datasets/cifar10-bin/";
const string g_cifar10_train = "data_all.bin";
const string g_cifar10_test = "test_batch.bin";
/// Main function
int main(int argc, char *argv[]) {
  lbann_comm *comm = initialize(argc, argv, 42);

#ifdef EL_USE_CUBLAS
  El::GemmUseGPU(32,32,32);
#endif

  try {

    //determine if we're going to scale, subtract mean, etc;
    //scaling/standardization is on a per-example basis (computed independantly
    //for each image)
    bool scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
    bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", false);
    bool unit_variance = Input("--unit-variance", "standardize to unit-variance", false);

    //if set to true, above three settings have no effect
    bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////

    // Initialize parameter defaults
    TrainingParams trainParams;
    trainParams.EpochCount = 20;
    trainParams.MBSize = 128;
    trainParams.LearnRate = 0.01;
    //trainParams.DropOut = -1.0f;
    trainParams.DropOut = 0.8;
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
    // load training data (CIFAR10)
    ///////////////////////////////////////////////////////////////////
    if (comm->am_world_master()) {
      cout << endl << "USING cifar10_reader\n\n";
    }
    cifar10_reader cifar10_trainset(trainParams.MBSize, true);
    cifar10_trainset.set_firstN(false);
    cifar10_trainset.set_role("train");
    cifar10_trainset.set_master(comm->am_world_master());
    cifar10_trainset.set_file_dir(g_cifar10_dir);
    cifar10_trainset.set_data_filename(g_cifar10_train);
    cifar10_trainset.set_validation_percent(trainParams.PercentageValidationSamples);
    cifar10_trainset.load();

    cifar10_trainset.scale(scale);
    cifar10_trainset.subtract_mean(subtract_mean);
    cifar10_trainset.unit_variance(unit_variance);
    cifar10_trainset.z_score(z_score);

    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data (CIFAR10)
    ///////////////////////////////////////////////////////////////////
    cifar10_reader cifar10_validation_set(cifar10_trainset); // Clone the training set object
    cifar10_validation_set.set_role("validation");
    cifar10_validation_set.use_unused_index_set();

    cout << "Num Neurons CIFAR10 " << cifar10_trainset.get_linearized_data_size() << endl;
    if (comm->am_world_master()) {
      size_t num_train = cifar10_trainset.get_num_data();
      size_t num_validate = cifar10_trainset.get_num_data();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      cout << "Num Neurons CIFAR10 " << cifar10_trainset.get_linearized_data_size() << endl;
      cout << "Training using " << train_percent << "% of the training data set, which is " << cifar10_trainset.get_num_data() << " samples." << endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << cifar10_validation_set.get_num_data() << " samples." << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data (CIFAR10)
    ///////////////////////////////////////////////////////////////////
    cifar10_reader cifar10_testset(trainParams.MBSize, true);
    cifar10_testset.set_firstN(false);
    cifar10_testset.set_role("test");
    cifar10_testset.set_master(comm->am_world_master());
    cifar10_testset.set_file_dir(g_cifar10_dir);
    cifar10_testset.set_data_filename(g_cifar10_test);
    cifar10_testset.set_use_percent(trainParams.PercentageTestingSamples);
    cifar10_testset.load();

    if (comm->am_world_master()) {
      cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << cifar10_testset.get_num_data() << " samples." << endl;
    }

    cifar10_testset.scale(scale);
    cifar10_testset.subtract_mean(subtract_mean);
    cifar10_testset.unit_variance(unit_variance);
    cifar10_testset.z_score(z_score);
    if (comm->am_world_master()) {
      cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << cifar10_testset.get_num_data() << " samples." << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////

    // Initialize optimizer
    optimizer_factory *optimizer_fac;
    if (trainParams.LearnRateMethod == 1) { // Adagrad
      optimizer_fac = new adagrad_factory(comm, trainParams.LearnRate);
      cout << "XX adagrad\n";
    } else if (trainParams.LearnRateMethod == 2) { // RMSprop
      optimizer_fac = new rmsprop_factory(comm, trainParams.LearnRate);
      cout << "XX rmsprop\n";
    } else if (trainParams.LearnRateMethod == 3) { // Adam
      optimizer_fac = new adam_factory(comm, trainParams.LearnRate);
      cout << "XX adam\n";
    } else {
      optimizer_fac = new sgd_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
      cout << "XX sgd\n";
    }

    // Initialize network
    deep_neural_network dnn(trainParams.MBSize, comm, new objective_functions::mean_squared_error(), optimizer_fac);
    std::map<execution_mode, generic_data_reader *> data_readers = {std::make_pair(execution_mode::training,&cifar10_trainset),
                                                           std::make_pair(execution_mode::validation, &cifar10_validation_set),
                                                           std::make_pair(execution_mode::testing, &cifar10_testset)
                                                          };


    Layer *input_layer = new input_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(comm, parallel_io, data_readers);
    dnn.add(input_layer);

    Layer *encode1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
                       1, comm,
                       1000, 
                       weight_initialization::glorot_uniform,
                       optimizer_fac->create_optimizer());
    dnn.add(encode1);
    

    Layer *relu1 = new relu_layer<data_layout::MODEL_PARALLEL>(2, comm);
    dnn.add(relu1);

    Layer *dropout1 = new dropout<data_layout::MODEL_PARALLEL>(3, 
                                               comm,
                                               trainParams.DropOut);
    dnn.add(dropout1);


    Layer *decode1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
                       4, comm,
                       cifar10_trainset.get_linearized_data_size(),
                       weight_initialization::glorot_uniform,
                       optimizer_fac->create_optimizer());
    dnn.add(decode1);
    
    Layer *relu2 = new sigmoid_layer<data_layout::MODEL_PARALLEL>(5, comm);
    dnn.add(relu2);

    Layer *dropout2 = new dropout<data_layout::MODEL_PARALLEL>(6,
                                               comm,
                                               trainParams.DropOut);
    dnn.add(dropout2);


    Layer* rcl  = new reconstruction_layer<data_layout::MODEL_PARALLEL>(7, comm, 
                                                          input_layer);
    dnn.add(rcl);

    
    lbann_callback_print print_cb;
    dnn.add_callback(&print_cb);

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

    if (comm->am_world_master()) {
      std::cout << "Parameter settings:" << std::endl;
      std::cout << "\tMini-batch size: " << trainParams.MBSize << std::endl;
      std::cout << "\tLearning rate: " << trainParams.LearnRate << std::endl << std::endl;
      std::cout << "\tEpoch count: " << trainParams.EpochCount << std::endl;
    }

    if (comm->am_world_master()) {
      optimizer *o = optimizer_fac->create_optimizer();
      cout << "\nOptimizer:\n" << o->get_description() << endl << endl;
      delete o;
      std::vector<Layer *>& layers = dnn.get_layers();
      for (size_t h=0; h<layers.size(); h++) {
        std::cout << h << " " << layers[h]->get_description() << endl;
      }
    }

    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////

    // Initialize the model's data structures
    dnn.setup();

    // set checkpoint directory and checkpoint interval
    dnn.set_checkpoint_dir(trainParams.ParameterDir);
    dnn.set_checkpoint_epochs(trainParams.CkptEpochs);
    dnn.set_checkpoint_steps(trainParams.CkptSteps);
    dnn.set_checkpoint_secs(trainParams.CkptSecs);

    // restart model from checkpoint if we have one
    dnn.restartShared();

    // train/test
    dnn.train(trainParams.EpochCount);
    
    // testing
    dnn.evaluate(execution_mode::testing);

    // Free dynamically allocated memory
    // delete target_layer;  // Causes segfault
    // delete input_layer;  // Causes segfault
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
    delete optimizer_fac;
  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);  /// Elemental exceptions
  }

  finalize(comm);

  return 0;
}
