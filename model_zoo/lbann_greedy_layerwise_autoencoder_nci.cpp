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
// lbann_dnn_nci.cpp - Autoencoder application for NCI
////////////////////////////////////////////////////////////////////////////////
#include "lbann/data_readers/lbann_data_reader_nci.hpp"
#include "lbann/callbacks/lbann_callback_dump_weights.hpp"
#include "lbann/callbacks/lbann_callback_dump_activations.hpp"
#include "lbann/callbacks/lbann_callback_dump_gradients.hpp"
#include "lbann/lbann.hpp"

using namespace std;
using namespace lbann;
#ifdef __LIB_ELEMENTAL
using namespace El;
#endif


//@todo use param options

int main(int argc, char* argv[])
{
    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);
    init_random(42);
    init_data_seq_random(42);
    lbann_comm* comm = NULL;

  try {


        ///////////////////////////////////////////////////////////////////
        // initalize grid, block
        ///////////////////////////////////////////////////////////////////
      TrainingParams trainParams;
      trainParams.DatasetRootDir = "/usr/mic/post1/metagenomics/cancer/anl_datasets/tmp_norm/";
      //trainParams.DumpWeights = "false"; //set to true to dump weight bias matrices
      //trainParams.DumpDir = "."; //provide directory to dump weight bias matrices
      trainParams.EpochCount = 10;
      trainParams.MBSize = 50;
      trainParams.LearnRate = 0.0001;
      trainParams.DropOut = -1.0f;
      trainParams.ProcsPerModel = 0;
      trainParams.PercentageTrainingSamples = 0.90;
      trainParams.PercentageValidationSamples = 1.00;
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
      lbann_comm* comm = new lbann_comm(trainParams.ProcsPerModel);
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
      }else {
        cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
      }

        ///////////////////////////////////////////////////////////////////
        // load training data
        ///////////////////////////////////////////////////////////////////
        //data_reader_nci nci_dataset(g_MBSize, true, grid.Rank()*g_MBSize, parallel_io*g_MBSize);
      clock_t load_time = clock();
      data_reader_nci nci_trainset(trainParams.MBSize, true);
      nci_trainset.set_data_filename(train_data);
      nci_trainset.set_use_percent(trainParams.PercentageTrainingSamples);
      nci_trainset.load();

      if (comm->am_world_master()) {
        cout << "Training using " << (trainParams.PercentageTrainingSamples*100) << "% of the training data set, which is " << nci_trainset.getNumData() << " samples." << endl;
      }

      ///////////////////////////////////////////////////////////////////
      // create a validation set from the unused training data (NCI)
      ///////////////////////////////////////////////////////////////////
      data_reader_nci nci_validation_set(nci_trainset); // Clone the training set object
      if (!nci_validation_set.swap_used_and_unused_index_sets()) { // Swap the used and unused index sets so that it validates on the remaining data
        if (comm->am_world_master()) {
          cout << "NCI validation data error" << endl;
        }
        return -1;
      }

      if(trainParams.PercentageValidationSamples == 1.00) {
        if (comm->am_world_master()) {
          cout << "Validating training using " << ((1.00 - trainParams.PercentageTrainingSamples)*100) << "% of the training data set, which is " << nci_validation_set.getNumData() << " samples." << endl;
        }
      }else {
        size_t preliminary_validation_set_size = nci_validation_set.getNumData();
        size_t final_validation_set_size = nci_validation_set.trim_data_set(trainParams.PercentageValidationSamples);
        if (comm->am_world_master()) {
          cout << "Trim the validation data set from " << preliminary_validation_set_size << " samples to " << final_validation_set_size << " samples." << endl;
        }
      }


        ///////////////////////////////////////////////////////////////////
        // load testing data (MNIST)
        ///////////////////////////////////////////////////////////////////
      data_reader_nci nci_testset(trainParams.MBSize, true);
      nci_testset.set_data_filename(test_data);
      nci_testset.load();

      if (comm->am_world_master()) cout << "Load Time " << ((double)clock() - load_time) / CLOCKS_PER_SEC << endl;


        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////
      Optimizer_factory *optimizer; //@todo replace with factory
      if (trainParams.LearnRateMethod == 1) { // Adagrad
        optimizer = new Adagrad_factory(comm, trainParams.LearnRate);
      }else if (trainParams.LearnRateMethod == 2) { // RMSprop
        optimizer = new RMSprop_factory(comm/*, trainParams.LearnRate*/);
      }else {
        optimizer = new SGD_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
      }
      layer_factory* lfac = new layer_factory();
      greedy_layerwise_autoencoder gla(trainParams.MBSize, comm, new objective_functions::mean_squared_error(comm), lfac, optimizer);

      std::map<execution_mode, DataReader*> data_readers = {std::make_pair(execution_mode::training,&nci_trainset),
                                                             std::make_pair(execution_mode::validation, &nci_validation_set),
                                                             std::make_pair(execution_mode::testing, &nci_testset)};

      input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(data_layout::DATA_PARALLEL, comm, parallel_io,
                                (int) trainParams.MBSize, data_readers);
      gla.add(input_layer);

      gla.add("FullyConnected", data_layout::MODEL_PARALLEL, 500, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(comm, trainParams.DropOut)});
      gla.add("FullyConnected", data_layout::MODEL_PARALLEL, 300, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(comm, trainParams.DropOut)});

      gla.add("FullyConnected", data_layout::MODEL_PARALLEL, 100, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(comm, trainParams.DropOut)});


      //Dump Weight-Bias matrices to files in DumpDir
      lbann_callback_dump_weights* dump_weights_cb;
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

      if (comm->am_world_master()) cout << "(Pre) train autoencoder - unsupersived training" << endl;
      gla.train(trainParams.EpochCount,true);

      if (trainParams.DumpWeights) {
        delete dump_weights_cb;
      }

      delete optimizer;
      delete comm;
    }
    catch (exception& e) { ReportException(e); }

    // free all resources by El and MPI
    Finalize();

    return 0;
}
