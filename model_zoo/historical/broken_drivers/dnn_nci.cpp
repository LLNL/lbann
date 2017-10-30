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
// dnn_nci.cpp - DNN application for NCI
////////////////////////////////////////////////////////////////////////////////
#include "lbann/data_readers/data_reader_nci.hpp"
#include "lbann/lbann.hpp"

using namespace std;
using namespace lbann;
#ifdef __LIB_ELEMENTAL
using namespace El;
#endif


//@todo use param options

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
    trainParams.DatasetRootDir = "/usr/mic/post1/metagenomics/cancer/anl_datasets/tmp_norm/";
    //trainParams.DumpWeights = "false"; //set to true to dump weight bias matrices
    //trainParams.DumpDir = "."; //provide directory to dump weight bias matrices
    trainParams.EpochCount = 2;
    trainParams.MBSize = 50;
    trainParams.LearnRate = 0.0001;
    trainParams.DropOut = -1.0f;
    trainParams.ProcsPerModel = 0;
    trainParams.PercentageTrainingSamples = 1.0;
    trainParams.PercentageValidationSamples = 0.1;
    PerformanceParams perfParams;
    perfParams.BlockSize = 256;

    // Parse command-line inputs
    trainParams.parse_params();
    perfParams.parse_params();

    // Read in the user specified network topology
    NetworkParams netParams;

    ProcessInput();
    PrintInputReport();

    // set algorithmic blocksize
    SetBlocksize(perfParams.BlockSize);

    const string train_data = trainParams.DatasetRootDir + trainParams.TrainFile;
    const string test_data  = trainParams.DatasetRootDir + trainParams.TestFile;

    // Set up the communicator and get the grid.
    comm = new lbann_comm(trainParams.ProcsPerModel);
    Grid& grid = comm->get_model_grid();
    if (comm->am_world_master()) {
      cout << "Number of models: " << comm->get_num_models() << endl;
      cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
      cout << "Train Data: " << train_data << endl;
      cout << "Test Data: " << test_data << endl;
      cout << endl;
    }

    int parallel_io = perfParams.MaxParIOSize;
    if(parallel_io == 0) {
      cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
      parallel_io = grid.Size();
    } else {
      cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // load training data
    ///////////////////////////////////////////////////////////////////
    //data_reader_nci nci_dataset(g_MBSize, true, grid.Rank()*g_MBSize, parallel_io*g_MBSize);
    clock_t load_time = clock();
    data_reader_nci nci_trainset(trainParams.MBSize, true);
    nci_trainset.set_data_filename(train_data);
    nci_trainset.set_validation_percent(trainParams.PercentageValidationSamples);
    nci_trainset.load();


    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data (NCI)
    ///////////////////////////////////////////////////////////////////
    data_reader_nci nci_validation_set(nci_trainset); // Clone the training set object
    nci_validation_set.use_unused_index_set();

    if (comm->am_world_master()) {
    /*
      size_t num_train = nci_trainset.getNumData();
      size_t num_validate = nci_trainset.getNumData();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      cout << "Training using " << train_percent << "% of the training data set, which is " << nci_trainset.getNumData() << " samples." << endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << nci_validation_set.getNumData() << " samples." << endl;
           */
    }


    ///////////////////////////////////////////////////////////////////
    // load testing data (MNIST)
    ///////////////////////////////////////////////////////////////////
    data_reader_nci nci_testset(trainParams.MBSize, true);
    nci_testset.set_data_filename(test_data);
    nci_testset.load();

    if (comm->am_world_master()) {
      cout << "Load Time " << ((double)clock() - load_time) / CLOCKS_PER_SEC << endl;
    }

    //nci_testset.setup(-1);



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
    deep_neural_network dnn(trainParams.MBSize, comm, new objective_functions::categorical_cross_entropy(comm), lfac, optimizer_fac);
    metrics::categorical_accuracy acc(data_layout::MODEL_PARALLEL, comm);
    dnn.add_metric(&acc);
    std::map<execution_mode, generic_data_reader *> data_readers = {std::make_pair(execution_mode::training,&nci_trainset),
                                                           std::make_pair(execution_mode::validation, &nci_validation_set),
                                                           std::make_pair(execution_mode::testing, &nci_testset)
                                                          };

    input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io,
                                                                                 data_readers);
    dnn.add(input_layer);

    dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, 500, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
    dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, 300, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
    dnn.add("softmax", data_layout::MODEL_PARALLEL, 2, activation_type::ID, weight_initialization::glorot_uniform, {});


    target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io,
                                                                                    data_readers, true);
    dnn.add(target_layer);

    //lbann_summary summarizer("/p/lscratchf/jacobs32", comm);
    // Print out information for each epoch.
    lbann_callback_print print_cb;
    dnn.add_callback(&print_cb);

    //Dump Weight-Bias matrices to files in DumpDir
    lbann_callback_dump_weights *dump_weights_cb;
    if (trainParams.DumpWeights) {
      dump_weights_cb = new lbann_callback_dump_weights(
        trainParams.DumpDir);
      dnn.add_callback(dump_weights_cb);
    }
    // Record training time information.
    //lbann_callback_timer timer_cb(&summarizer);
    //dnn.add_callback(&timer_cb);
    // Summarize information to Tensorboard.
    //lbann_callback_summary summary_cb(&summarizer, 25);
    //dnn.add_callback(&summary_cb);

    if (comm->am_world_master()) {
      cout << "Layer initialized:" << endl;
      cout << "Print Layers using factory ";
      lfac->print();
      cout << endl;
    }

    if (comm->am_world_master()) {
      cout << "Parameter settings:" << endl;
      cout << "\tMini-batch size: " << trainParams.MBSize << endl;
      cout << "\tLearning rate: " << trainParams.LearnRate << endl;
      cout << "\tEpoch count: " << trainParams.EpochCount << endl;
      cout << "\tDump Weights? " << trainParams.DumpWeights << endl;
      cout << "\tDump Dir : " << trainParams.DumpDir << endl;
    }

    if (comm->am_world_master()) {
      optimizer *o = optimizer_fac->create_optimizer();
      cout << "\nOptimizer:\n" << o->get_description() << endl << endl;
      delete o;
      std::vector<Layer *>& layers = model->get_layers();
      for (size_t h=0; h<layers.size(); h++) {
        std::cout << h << " " << layers[h]->get_description() << endl;
      }
    }

    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////

    // Initialize the model's data structures
    dnn.setup();

    //train/test
    for(int t = 0; t < trainParams.EpochCount; t++) {
      dnn.train(1,true);
      // testing
      dnn.evaluate(execution_mode::testing);
    }

    if (trainParams.DumpWeights) {
      delete dump_weights_cb;
    }

    delete optimizer_fac;
  } catch (exception& e) {
    ReportException(e);
  }

  // free all resources by El and MPI
  delete comm;
  Finalize();

  return 0;
}
