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


#if 0
//the following options have default values (see below), but can be
//over-ridden on the cmd line:
--max_par_io_size=<int>
--block_size=<int>
--procs_per_model=<int>
--mb_size=<int>
--training_samples=<int>
--testing_samples=<int>
--network=<string>
--percentage_validation_samples=<double>
--learn_method=<string> //adagrad, rmsprop, sgd, adam, hypergradient_adam
--learn_rate=<double>
--decay_rate=<double>
--momentum=<double>
--epoch_count=<int>
--dropout=<float>
#endif


int main(int argc, char *argv[]) {
  lbann_comm *comm = initialize(argc, argv, 42);
  bool master = comm->am_world_master();

  options *opts = options::get();
  opts->init(argc, argv);

  try {

    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////

    // set algorithmic blocksize
    SetBlocksize( opts->get_int("block_size", 256) );

    // Set up the communicator and get the grid.
    comm->split_models( opts->get_int("procs_per_model", 0) );
    Grid& grid = comm->get_model_grid();
    if (master) {
      cout << "Number of models: " << comm->get_num_models() << endl;
      cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
      cout << endl;
    }

    int parallel_io = (opts->get_int("max_par_io_size", 0));
    if(parallel_io == 0) {
      if (master) {
        cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
      }  
      parallel_io = grid.Size();
    } else {
      if (master) {
        cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
      }
    }

    ///////////////////////////////////////////////////////////////////
    // load training data
    ///////////////////////////////////////////////////////////////////
    // clock_t load_time = clock();
    int mb_size = opts->get_int("mb_size", 192);
    int training_samples = opts->get_int("training_samples", 1024);
    int testing_samples = opts->get_int("testing_samples", 256);
    int num_features = opts->get_int("num_features", 1000);
    double percentage_validation_samples = 0.5;

    data_reader_synthetic synthetic_trainset(mb_size, training_samples, num_features);
    synthetic_trainset.set_validation_percent(percentage_validation_samples);
    synthetic_trainset.load();

    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data
    ///////////////////////////////////////////////////////////////////
    data_reader_synthetic synthetic_validation_set(synthetic_trainset); // Clone the training set object
    synthetic_validation_set.use_unused_index_set();

    if (master) {
      size_t num_train = synthetic_trainset.get_num_data();
      size_t num_validate = synthetic_trainset.get_num_data();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      cout << "Training using " << train_percent << "% of the training data set, which is " << synthetic_trainset.get_num_data() << " samples." << endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << synthetic_validation_set.get_num_data() << " samples." << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data
    ///////////////////////////////////////////////////////////////////
    data_reader_synthetic synthetic_testset(mb_size, testing_samples, num_features);
    synthetic_testset.load();

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////
    optimizer_factory *optimizer_fac;
    string learn_method = opts->get_string("learn_method", "rmsprop");

    //dah - hopefully these will be cast correctly ...
    DataType learn_rate = opts->get_double("learn_rate", 0.0001);
    DataType decay_rate = opts->get_double("decay_rate", 0.5);
    DataType momentum = opts->get_double("momentum", 0.9);

    if (learn_method == "adagrad") {
      optimizer_fac = new adagrad_factory(comm, learn_rate);
    } else if (learn_method == "rmsprop") {
      optimizer_fac = new rmsprop_factory(comm, learn_rate);
    } else if (learn_method == "adam") {
      optimizer_fac = new adam_factory(comm, learn_rate);
    } else if (learn_method == "hypergradient_adam") {
      optimizer_fac = new hypergradient_adam_factory(comm, learn_rate);
    } else if (learn_method == "sgd_factory") {
      optimizer_fac = new sgd_factory(comm, learn_rate, momentum, decay_rate, true);
    } else {
      stringstream err;
      err << __FILE__ << " " << __LINE__
          << " :: unknown value for learn_method: " << learn_method << "; must be one of "
          << "adagrad, rmsprop, sgd, adam, hypergradient_adam";
      throw lbann_exception(err.str());
    }

    // Initialize network
    deep_neural_network dnn(mb_size, comm, new objective_functions::mean_squared_error(),optimizer_fac);
    std::map<execution_mode, generic_data_reader *> data_readers = {std::make_pair(execution_mode::training,&synthetic_trainset),
                                                           std::make_pair(execution_mode::validation, &synthetic_validation_set),
                                                           std::make_pair(execution_mode::testing, &synthetic_testset)
                                                          };

    float drop_out = opts->get_float("dropout", -1.0);

    Layer *input_layer = new input_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(comm, parallel_io, data_readers);
    dnn.add(input_layer);

    Layer *encode1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
                       1, comm,
                       100, 
                       weight_initialization::glorot_uniform,
                       optimizer_fac->create_optimizer());
    dnn.add(encode1);
    

    Layer *relu1 = new relu_layer<data_layout::MODEL_PARALLEL>(2, comm);
    dnn.add(relu1);

    Layer *dropout1 = new dropout<data_layout::MODEL_PARALLEL>(3, 
                                               comm,
                                               drop_out);
    dnn.add(dropout1);


    Layer *decode1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
                       4, comm,
                       synthetic_trainset.get_linearized_data_size(),
                       weight_initialization::glorot_uniform,
                       optimizer_fac->create_optimizer());
    dnn.add(decode1);
    
    Layer *relu2 = new sigmoid_layer<data_layout::MODEL_PARALLEL>(5, comm);
    dnn.add(relu2);

    Layer *dropout2 = new dropout<data_layout::MODEL_PARALLEL>(6,
                                               comm,
                                               drop_out);
    dnn.add(dropout2);

    Layer* rcl  = new reconstruction_layer<data_layout::MODEL_PARALLEL>(7, comm, 
                                                          input_layer);
    dnn.add(rcl);

    int epoch_count = opts->get_int("epoch_count", 12);
    
    lbann_callback_print print_cb;
    dnn.add_callback(&print_cb);
    lbann_callback_check_reconstruction_error cre;
    dnn.add_callback(&cre);
    if (master) {
      cout << "Parameter settings:" << endl;
      cout << "\tTraining sample size: " << training_samples << endl;
      cout << "\tTesting sample size: " << testing_samples << endl;
      cout << "\tFeature vector size: " << num_features << endl;
      cout << "\tMini-batch size: " << mb_size << endl;
      cout << "\tLearning rate: " << learn_rate << endl;
      cout << "\tEpoch count: " << epoch_count << endl;
    }

    dnn.setup();
    
    if(master) std::cout << "Auto Testing: "
        << "Reconstruction error should gradually decrease to below 1 at around 8th epoch" 
        <<  "for a successful testing"  << std::endl;
    while (dnn.get_cur_epoch() < epoch_count) {
      dnn.train(1);
    }

    if(master) std::cout << "TEST FAILED " << std::endl;
 
    delete optimizer_fac;
    delete comm;
  } catch (exception& e) {
    ReportException(e);
  }

  if (master) {
    opts->print();
  }

  // free all resources by El and MPI
  Finalize();

  return 0;
}
