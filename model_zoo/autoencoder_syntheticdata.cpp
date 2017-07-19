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
#include "lbann/lbann.hpp"

using namespace std;
using namespace lbann;
#ifdef __LIB_ELEMENTAL
using namespace El;
#endif


//@todo use param options

int main(int argc, char *argv[]) {
  lbann_comm *comm = initialize(argc, argv, 42);

  try {


    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////
    TrainingParams trainParams;
    trainParams.LearnRate = 0.0001;
    trainParams.DropOut = -1.0f;
    trainParams.ProcsPerModel = 0;
    trainParams.PercentageTrainingSamples = 1.0;
    trainParams.PercentageValidationSamples = 0.5;
    PerformanceParams perfParams;
    perfParams.BlockSize = 256;

    // Parse command-line inputs
    trainParams.parse_params();
    perfParams.parse_params();

    // Read in the user specified network topology
    NetworkParams netParams;
    netParams.parse_params();

    ProcessInput();
    PrintInputReport();

    // set algorithmic blocksize
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
    if(parallel_io == 0) {
      cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
      parallel_io = grid.Size();
    } else {
      cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // load training data
    ///////////////////////////////////////////////////////////////////
    clock_t load_time = clock();
    data_reader_synthetic synthetic_trainset(trainParams.MBSize, trainParams.TrainingSamples, netParams.Network[0]);
    synthetic_trainset.set_validation_percent(trainParams.PercentageValidationSamples);
    synthetic_trainset.load();


    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data
    ///////////////////////////////////////////////////////////////////
    data_reader_synthetic synthetic_validation_set(synthetic_trainset); // Clone the training set object
    synthetic_validation_set.use_unused_index_set();


    if (comm->am_world_master()) {
      size_t num_train = synthetic_trainset.getNumData();
      size_t num_validate = synthetic_trainset.getNumData();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      cout << "Training using " << train_percent << "% of the training data set, which is " << synthetic_trainset.getNumData() << " samples." << endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << synthetic_validation_set.getNumData() << " samples." << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data
    ///////////////////////////////////////////////////////////////////
    data_reader_synthetic synthetic_testset(trainParams.MBSize, trainParams.TestingSamples,netParams.Network[0]);
    synthetic_testset.load();

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////
    optimizer_factory *optimizer_fac;
    if (trainParams.LearnRateMethod == 1) { // Adagrad
      optimizer_fac = new adagrad_factory(comm, trainParams.LearnRate);
      if(comm->am_world_master()) cout << "XX adagrad\n";
    } else if (trainParams.LearnRateMethod == 2) { // RMSprop
      optimizer_fac = new rmsprop_factory(comm, trainParams.LearnRate);
      if(comm->am_world_master()) cout << "XX rmsprop\n";
    } else if (trainParams.LearnRateMethod == 3) { // Adam
      optimizer_fac = new adam_factory(comm, trainParams.LearnRate);
      if(comm->am_world_master()) cout << "XX adam\n";
    } else {
      optimizer_fac = new sgd_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
      if(comm->am_world_master()) cout << "XX sgd\n";
    }

    // Initialize network
    deep_neural_network dnn(trainParams.MBSize, comm, new objective_functions::mean_squared_error(comm),optimizer_fac);
    std::map<execution_mode, generic_data_reader *> data_readers = {std::make_pair(execution_mode::training,&synthetic_trainset),
                                                           std::make_pair(execution_mode::validation, &synthetic_validation_set),
                                                           std::make_pair(execution_mode::testing, &synthetic_testset)
                                                          };


    Layer *input_layer = new input_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(comm, trainParams.MBSize, parallel_io, data_readers);
    dnn.add(input_layer);

    Layer *encode1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
                       1, comm, trainParams.MBSize,
                       100, 
                       weight_initialization::glorot_uniform,
                       optimizer_fac->create_optimizer());
    dnn.add(encode1);
    

    Layer *relu1 = new relu_layer<data_layout::MODEL_PARALLEL>(2, comm,
                                               trainParams.MBSize);
    dnn.add(relu1);

    Layer *dropout1 = new dropout<data_layout::MODEL_PARALLEL>(3, 
                                               comm, trainParams.MBSize,
                                               trainParams.DropOut);
    dnn.add(dropout1);


    Layer *decode1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
                       4, comm, trainParams.MBSize,
                       synthetic_trainset.get_linearized_data_size(),
                       weight_initialization::glorot_uniform,
                       optimizer_fac->create_optimizer());
    dnn.add(decode1);
    
    Layer *relu2 = new sigmoid_layer<data_layout::MODEL_PARALLEL>(5, comm,
                                               trainParams.MBSize);
    dnn.add(relu2);

    Layer *dropout2 = new dropout<data_layout::MODEL_PARALLEL>(6,
                                               comm, trainParams.MBSize,
                                               trainParams.DropOut);

    Layer* rcl  = new reconstruction_layer<data_layout::MODEL_PARALLEL>(7, comm, 
                                                          trainParams.MBSize, input_layer);
    dnn.add(rcl);

    
    lbann_callback_print print_cb;
    dnn.add_callback(&print_cb);
    lbann_callback_check_reconstruction_error cre;
    dnn.add_callback(&cre);
    if (comm->am_world_master()) {
      cout << "Parameter settings:" << endl;
      cout << "\tTraining sample size: " << trainParams.TrainingSamples << endl;
      cout << "\tTesting sample size: " << trainParams.TestingSamples << endl;
      cout << "\tFeature vector size: " << netParams.Network[0] << endl;
      cout << "\tMini-batch size: " << trainParams.MBSize << endl;
      cout << "\tLearning rate: " << trainParams.LearnRate << endl;
      cout << "\tEpoch count: " << trainParams.EpochCount << endl;
    }



    dnn.setup();

    while (dnn.get_cur_epoch() < trainParams.EpochCount) {
      dnn.train(1);
    }

    if(comm->am_world_master()) std::cout << "TEST FAILED " << std::endl;
 
    delete optimizer_fac;
    delete comm;
  } catch (exception& e) {
    ReportException(e);
  }

  // free all resources by El and MPI
  Finalize();

  return 0;
}
