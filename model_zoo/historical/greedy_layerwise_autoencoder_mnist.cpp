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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_mnist.hpp"
#include "lbann/callbacks/callback_save_images.hpp"
#include "lbann/callbacks/callback_summary.hpp"
#include "lbann/lbann.hpp"

using namespace std;
using namespace lbann;
using namespace El;


// layer definition
/*const std::vector<int> g_LayerDim = {784, 100, 30, 10};
const uint g_NumLayers = g_LayerDim.size(); // # layers*/

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

    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////

    // Initialize parameter defaults
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/MNIST/";
    trainParams.EpochCount = 50;
    trainParams.MBSize = 192;
    trainParams.DumpWeights=0;
    trainParams.DumpActivations=0;
    trainParams.DumpGradients=0;
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

    ///////////////////////////////////////////////////////////////////
    // load training data (MNIST)
    ///////////////////////////////////////////////////////////////////
    mnist_reader mnist_trainset(trainParams.MBSize, true);
    mnist_trainset.set_file_dir(trainParams.DatasetRootDir);
    mnist_trainset.set_data_filename(g_MNIST_TrainImageFile);
    mnist_trainset.set_label_filename(g_MNIST_TrainLabelFile);
    mnist_trainset.set_validation_percent(trainParams.PercentageValidationSamples);
    mnist_trainset.load();

    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data (MNIST)
    ///////////////////////////////////////////////////////////////////
    mnist_reader mnist_validation_set(mnist_trainset); // Clone the training set object
    mnist_validation_set.use_unused_index_set();
    if (comm->am_world_master()) {
      size_t num_train = mnist_trainset.get_num_data();
      size_t num_validate = mnist_trainset.get_num_data();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      cout << "Training using " << train_percent << "% of the training data set, which is " << mnist_trainset.get_num_data() << " samples." << endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << mnist_validation_set.get_num_data() << " samples." << endl;
    }


    ///////////////////////////////////////////////////////////////////
    // load testing data (MNIST)
    ///////////////////////////////////////////////////////////////////
    mnist_reader mnist_testset(trainParams.MBSize, true);
    mnist_testset.set_file_dir(trainParams.DatasetRootDir);
    mnist_testset.set_data_filename(g_MNIST_TestImageFile);
    mnist_testset.set_label_filename(g_MNIST_TestLabelFile);
    mnist_testset.set_use_percent(trainParams.PercentageTestingSamples);
    mnist_testset.load();
    if (comm->am_world_master()) {
      cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << mnist_testset.get_num_data() << " samples." << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////

    // Initialize optimizer
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

    // Initialize network
    greedy_layerwise_autoencoder gla(trainParams.MBSize, comm, new objective_functions::mean_squared_error(), optimizer_fac);
    std::map<execution_mode, generic_data_reader *> data_readers = {std::make_pair(execution_mode::training,&mnist_trainset),
                                                           std::make_pair(execution_mode::validation, &mnist_validation_set),
                                                           std::make_pair(execution_mode::testing, &mnist_testset)
                                                          };

    Layer *input_layer = new input_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(comm, parallel_io, data_readers);
    gla.add(input_layer);

    Layer *encode1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
                       1, comm,
                       100, 
                       weight_initialization::glorot_uniform,
                       optimizer_fac->create_optimizer());
    gla.add(encode1);
    
    Layer *relu1 = new sigmoid_layer<data_layout::MODEL_PARALLEL>(2, comm);
    gla.add(relu1);


    Layer *decode1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
                       3, comm,
                       mnist_trainset.get_linearized_data_size(),
                       weight_initialization::glorot_uniform,
                       optimizer_fac->create_optimizer());
    gla.add(decode1);
    
    Layer *relu2 = new sigmoid_layer<data_layout::MODEL_PARALLEL>(4, comm);
    gla.add(relu2);


    Layer* rcl1  = new reconstruction_layer<data_layout::MODEL_PARALLEL>(5, comm, 
                                                          input_layer);
    gla.add(rcl1);

   // Laywerise2 
    Layer *encode2 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
                       6, comm,
                       50, 
                       weight_initialization::glorot_uniform,
                       optimizer_fac->create_optimizer());
    gla.add(encode2);
    

    Layer *relu3 = new relu_layer<data_layout::MODEL_PARALLEL>(7, comm);
    gla.add(relu3);


    Layer *decode2 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
                       8, comm,
                       100,
                       weight_initialization::glorot_uniform,
                       optimizer_fac->create_optimizer());
    gla.add(decode2);
    
    Layer *relu4 = new sigmoid_layer<data_layout::MODEL_PARALLEL>(9, comm);
    gla.add(relu4);


    Layer* rcl2  = new reconstruction_layer<data_layout::MODEL_PARALLEL>(10, comm, 
                                                          relu1);

    gla.add(rcl2);
    
    lbann_callback_dump_weights *dump_weights_cb = nullptr;
    lbann_callback_dump_activations *dump_activations_cb = nullptr;
    lbann_callback_dump_gradients *dump_gradients_cb = nullptr;
    if (trainParams.DumpWeights) {
      dump_weights_cb = new lbann_callback_dump_weights(
        trainParams.DumpDir,1000);
      gla.add_callback(dump_weights_cb);
    }
    if (trainParams.DumpActivations) {
      dump_activations_cb = new lbann_callback_dump_activations(
        trainParams.DumpDir,1000);
      gla.add_callback(dump_activations_cb);
    }
    if (trainParams.DumpGradients) {
      dump_gradients_cb = new lbann_callback_dump_gradients(
        trainParams.DumpDir,1000);
      gla.add_callback(dump_gradients_cb);
    }

    if (comm->am_world_master()) {
      cout << "Parameter settings:" << endl;
      cout << "\tMini-batch size: " << trainParams.MBSize << endl;
      cout << "\tLearning rate: " << trainParams.LearnRate << endl;
      cout << "\tEpoch count: " << trainParams.EpochCount << endl;
    }

    if (comm->am_world_master()) {
      optimizer *o = optimizer_fac->create_optimizer();
      cout << "\nOptimizer:\n" << o->get_description() << endl << endl;
      std::vector<Layer *>& layers = gla.get_layers();
      for (size_t h=0; h<layers.size(); h++) {
        std::cout << h << " " << layers[h]->get_description() << endl;
      }
    }

    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////

    // Initialize the model's data structures
    gla.setup();

    // set checkpoint directory and checkpoint interval
    // @TODO: add to lbann_proto
    gla.set_checkpoint_dir(trainParams.ParameterDir);
    gla.set_checkpoint_epochs(trainParams.CkptEpochs);
    gla.set_checkpoint_steps(trainParams.CkptSteps);
    gla.set_checkpoint_secs(trainParams.CkptSecs);

    // restart model from checkpoint if we have one
    gla.restartShared();

    if (comm->am_world_master()) {
      cout << "(Pre) train autoencoder - unsupersived training" << endl;
    }
    gla.train(trainParams.EpochCount);


    // Free dynamically allocated memory
    // delete target_layer;  // Causes segfault
    // delete input_layer;  // Causes segfault
    // delete lfac;  // Causes segfault
    delete optimizer_fac;
    delete comm;

  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);  /// Elemental exceptions
  }

  // free all resources by El and MPI
  Finalize();

  return 0;
}
