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
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/data_readers/data_reader_mnist.hpp"
#include "lbann/callbacks/callback_ltfb.hpp"

using namespace lbann;

int main(int argc, char *argv[]) {
  El::Initialize(argc, argv);
  init_random(42);  // Deterministic initialization across every model.
  init_data_seq_random(42);
  lbann_comm *comm = NULL;

  try {
    // Get data files.
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

    // Set up parameter defaults.
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/MNIST/";
    trainParams.EpochCount = 20;
    trainParams.MBSize = 10;
    trainParams.LearnRate = 0.0001;
    trainParams.DropOut = -1.0f;
    trainParams.ProcsPerModel = 2;  // Use one Catalyst node.
    trainParams.PercentageTrainingSamples = 1.0;
    trainParams.PercentageValidationSamples = 0.1;
    PerformanceParams perfParams;
    perfParams.BlockSize = 256;

    // Parse command-line inputs
    trainParams.parse_params();
    perfParams.parse_params();

    ProcessInput();
    PrintInputReport();

    // set algorithmic blocksize
    SetBlocksize(perfParams.BlockSize);

    // Set up the communicator and get the grid.
    comm = new lbann_comm(trainParams.ProcsPerModel);
    Grid& grid = comm->get_model_grid();
    if (comm->am_world_master()) {
      cout << "Number of models: " << comm->get_num_models() <<
           " (" << comm->get_procs_per_model() << " procs per model)" << endl;
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
    mnist_reader mnist_trainset(trainParams.MBSize);
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
      size_t num_train = mnist_trainset.getNumData();
      size_t num_validate = mnist_trainset.getNumData();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      cout << "Training using " << train_percent << "% of the training data set, which is " << mnist_trainset.getNumData() << " samples." << endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << mnist_validation_set.getNumData() << " samples." << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data (MNIST)
    ///////////////////////////////////////////////////////////////////
    mnist_reader mnist_testset(trainParams.MBSize);
    mnist_testset.set_file_dir(trainParams.DatasetRootDir);
    mnist_testset.set_data_filename(g_MNIST_TestImageFile);
    mnist_testset.set_label_filename(g_MNIST_TestLabelFile);
    mnist_testset.set_use_percent(trainParams.PercentageTestingSamples);
    mnist_testset.load();

    if (comm->am_world_master()) {
      cout << "Testing using " << (trainParams.PercentageTestingSamples*100) <<
           "% of the testing data set, which is " << mnist_testset.getNumData() <<
           " samples." << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////

    // Initialize optimizer
    optimizer_factory *optimizer_fac = new sgd_factory(
      comm, trainParams.LearnRate, 0, 0, false);

    layer_factory *lfac = new layer_factory();
    deep_neural_network dnn(trainParams.MBSize, comm, new objective_functions::categorical_cross_entropy(comm), lfac, optimizer_fac);
    metrics::categorical_accuracy acc(data_layout::MODEL_PARALLEL, comm);
    dnn.add_metric(&acc);
    std::map<execution_mode, generic_data_reader *> data_readers = {std::make_pair(execution_mode::training,&mnist_trainset),
                                                           std::make_pair(execution_mode::validation, &mnist_validation_set),
                                                           std::make_pair(execution_mode::testing, &mnist_testset)
                                                          };
    input_layer *input_layer_ = new input_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, trainParams.MBSize, data_readers);
    dnn.add(input_layer_);
    dnn.add(
      "FullyConnected", data_layout::MODEL_PARALLEL, 100,
      trainParams.ActivationType, weight_initialization::glorot_uniform,
      {});
    dnn.add(
      "FullyConnected", data_layout::MODEL_PARALLEL, 30,
      trainParams.ActivationType, weight_initialization::glorot_uniform,
      {});
    dnn.add(
      "softmax", data_layout::MODEL_PARALLEL, 10,
      activation_type::ID, weight_initialization::glorot_uniform, {});
    target_layer *target_layer_ = new target_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, trainParams.MBSize, data_readers, true);
    dnn.add(target_layer_);

    //lbann_summary summarizer(trainParams.SummaryDir, comm);
    // Print out information for each epoch.
    lbann_callback_print print_cb;
    dnn.add_callback(&print_cb);
    // Record training time information.
    lbann_callback_timer timer_cb;
    dnn.add_callback(&timer_cb);

    // Duplicate model.
    mnist_reader mnist_trainset2(trainParams.MBSize);
    mnist_trainset2.set_file_dir(trainParams.DatasetRootDir);
    mnist_trainset2.set_data_filename(g_MNIST_TrainImageFile);
    mnist_trainset2.set_label_filename(g_MNIST_TrainLabelFile);
    mnist_trainset2.set_validation_percent(trainParams.PercentageValidationSamples);
    mnist_trainset2.load();
    mnist_reader mnist_validation_set2(mnist_trainset2); // Clone the training set object
    mnist_validation_set2.use_unused_index_set();
    mnist_reader mnist_testset2(trainParams.MBSize);
    mnist_testset2.set_file_dir(trainParams.DatasetRootDir);
    mnist_testset2.set_data_filename(g_MNIST_TestImageFile);
    mnist_testset2.set_label_filename(g_MNIST_TestLabelFile);
    mnist_testset2.set_use_percent(trainParams.PercentageTestingSamples);
    mnist_testset2.load();
    optimizer_factory *optimizer_fac2 = new sgd_factory(
      comm, trainParams.LearnRate, 0, 0, false);
    layer_factory *lfac2 = new layer_factory();
    deep_neural_network dnn2(trainParams.MBSize, comm, new objective_functions::categorical_cross_entropy(comm), lfac2, optimizer_fac2);
    metrics::categorical_accuracy acc2(data_layout::MODEL_PARALLEL, comm);
    dnn2.add_metric(&acc2);
    std::map<execution_mode, generic_data_reader *> data_readers2 = {
      std::make_pair(execution_mode::training,&mnist_trainset2),
      std::make_pair(execution_mode::validation, &mnist_validation_set2),
      std::make_pair(execution_mode::testing, &mnist_testset2)
    };
    input_layer *input_layer2 = new input_layer_distributed_minibatch_parallel_io(
      data_layout::MODEL_PARALLEL, comm, parallel_io, trainParams.MBSize, data_readers2);
    dnn2.add(input_layer2);
    dnn2.add(
      "FullyConnected", data_layout::MODEL_PARALLEL, 100,
      trainParams.ActivationType, weight_initialization::glorot_uniform,
      {});
    dnn2.add(
      "FullyConnected", data_layout::MODEL_PARALLEL, 30,
      trainParams.ActivationType, weight_initialization::glorot_uniform,
      {});
    dnn2.add(
      "softmax", data_layout::MODEL_PARALLEL, 10,
      activation_type::ID, weight_initialization::glorot_uniform, {});
    target_layer *target_layer2 = new target_layer_distributed_minibatch_parallel_io(
      data_layout::MODEL_PARALLEL, comm, parallel_io, trainParams.MBSize, data_readers2, true);
    dnn2.add(target_layer2);

    // LTFB.
    lbann_callback_ltfb ltfb(45, &dnn2);
    dnn.add_callback(&ltfb);

    if (comm->am_world_master()) {
      cout << "Parameter settings:" << endl;
      cout << "\tMini-batch size: " << trainParams.MBSize << endl;
      cout << "\tLearning rate: " << trainParams.LearnRate << endl << endl;
      cout << "\tEpoch count: " << trainParams.EpochCount << endl;
    }

    comm->global_barrier();

    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////

    // Initialize the model's data structures
    dnn.setup();
    dnn2.setup();

    // Reinitialize the RNG differently for each rank.
    init_random(comm->get_rank_in_world() + 1);

    comm->global_barrier();

    // train/test
    for (int t = 0; t < trainParams.EpochCount; t++) {
      dnn.train(1, true);
      dnn.evaluate();
    }
  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);
  }

  // free all resources by El and MPI
  Finalize();

  return 0;
}
