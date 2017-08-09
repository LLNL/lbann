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
// dnn_nci.cpp - Autoencoder application for NCI
////////////////////////////////////////////////////////////////////////////////
#include "lbann/data_readers/data_reader_nci.hpp"
#include "lbann/callbacks/callback_dump_weights.hpp"
#include "lbann/callbacks/callback_dump_activations.hpp"
#include "lbann/callbacks/callback_dump_gradients.hpp"
#include "lbann/lbann.hpp"

using namespace std;
using namespace lbann;
#ifdef __LIB_ELEMENTAL
using namespace El;
#endif


//@todo use param options

int main(int argc, char *argv[]) {
  // El initialization (similar to MPI_Init)
  lbann_comm *comm = initialize(argc, argv, 42);

#ifdef EL_USE_CUBLAS
  El::GemmUseGPU(32,32,32);
#endif

  try {


    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/usr/mic/post1/metagenomics/cancer/anl_datasets/tmp_norm/";
    //trainParams.DumpWeights = "false"; //set to true to dump weight bias matrices
    //trainParams.DumpDir = "."; //provide directory to dump weight bias matrices
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
    comm->split_models(trainParams.ProcsPerModel);
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
      size_t num_train = nci_trainset.getNumData();
      size_t num_validate = nci_trainset.getNumData();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      cout << "Training using " << train_percent << "% of the training data set, which is " << nci_trainset.getNumData() << " samples." << endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << nci_validation_set.getNumData() << " samples." << endl;
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
    greedy_layerwise_autoencoder gla(trainParams.MBSize, comm, new objective_functions::mean_squared_error(comm), optimizer_fac);

    std::map<execution_mode, generic_data_reader *> data_readers = {std::make_pair(execution_mode::training,&nci_trainset),
                                                           std::make_pair(execution_mode::validation, &nci_validation_set),
                                                           std::make_pair(execution_mode::testing, &nci_testset)
                                                          };

    Layer *input_layer = new input_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(comm, parallel_io, data_readers);
    gla.add(input_layer);
    Layer *fc1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(1,
                                                        nci_trainset.get_linearized_data_size(), 500,trainParams.MBSize,
                                                        weight_initialization::glorot_uniform, comm, optimizer_fac->create_optimizer());
    gla.add(fc1);

    /*gla.add("FullyConnected", data_layout::MODEL_PARALLEL, 500, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
    gla.add("FullyConnected", data_layout::MODEL_PARALLEL, 300, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});

    gla.add("FullyConnected", data_layout::MODEL_PARALLEL, 100, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
*/

    //Dump Weight-Bias matrices to files in DumpDir
    lbann_callback_dump_weights *dump_weights_cb;
    if (trainParams.DumpWeights) {
      dump_weights_cb = new lbann_callback_dump_weights(
        trainParams.DumpDir);
      gla.add_callback(dump_weights_cb);
    }


    if (comm->am_world_master()) {
      cout << "Parameter settings:" << endl;
      cout << "\tMini-batch size: " << trainParams.MBSize << endl;
      cout << "\tLearning rate: " << trainParams.LearnRate << endl;
      cout << "\tEpoch count: " << trainParams.EpochCount << endl;
      cout << "\t Dump Weights? " << trainParams.DumpWeights << endl;
      cout << "\tDump Dir : " << trainParams.DumpDir << endl;
    }



    gla.setup();

    // set checkpoint directory and checkpoint interval
    // @TODO: add to lbann_proto
    gla.set_checkpoint_dir(trainParams.ParameterDir);
    gla.set_checkpoint_epochs(trainParams.CkptEpochs);
    gla.set_checkpoint_steps(trainParams.CkptSteps);
    gla.set_checkpoint_secs(trainParams.CkptSecs);

    // restart model from checkpoint if we have one
    gla.restartShared();

    for(int i =1; i <= trainParams.EpochCount; i++) {

      if (comm->am_world_master()) {
        std::cout << "\n(Pre) train autoencoder - unsupersived training, global epoch [ " << i << " ]" << std::endl;
        std::cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << std::endl;
      }
      gla.train(1,true);
      gla.reset_phase();
    }

    if (trainParams.DumpWeights) {
      delete dump_weights_cb;
    }

    delete optimizer_fac;
    delete comm;
  } catch (exception& e) {
    ReportException(e);
  }

  // free all resources by El and MPI
  Finalize();

  return 0;
}
