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
// lbann_dnn_nci.cpp - DNN application for NCI
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_nci_regression.hpp"
#include "lbann/lbann.hpp"

using namespace std;
using namespace lbann;
#ifdef __LIB_ELEMENTAL
using namespace El;
#endif

///////////////////////////////////////////////////////////////////
// Initialize parameter defaults
///////////////////////////////////////////////////////////////////
void init_params(TrainingParams& trainParams, PerformanceParams& perfParams)
{
  trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/cancer/anl_datasets/tmp_norm";
  trainParams.EpochCount = 2;
  trainParams.MBSize = 50;
  trainParams.LearnRateMethod = 1;
  trainParams.LearnRate = 0.0001;
  trainParams.LrDecayRate = 0.0;
  trainParams.LrDecayCycles = 5000;
  trainParams.LrMomentum = 0.9; // for SGD
  trainParams.ActivationType = lbann::activation_type::RELU;
  trainParams.DropOut = -1.0f; // default -1.0 which means 'no dropout'
  trainParams.WeightInitType = weight_initialization::glorot_uniform;
  trainParams.ProcsPerModel = 0;
  trainParams.PercentageTrainingSamples = 0.90;
  trainParams.PercentageValidationSamples = 1.00;
  perfParams.BlockSize = 256;

  // Parse command-line inputs
  trainParams.parse_params();
  perfParams.parse_params();

  ProcessInput();
  PrintInputReport();

  // Set algorithmic blocksize
  SetBlocksize(perfParams.BlockSize);

}

void print_params(lbann_comm* comm, TrainingParams& trainParams, PerformanceParams& perfParams)
{
  if (comm == NULL) return;

  Grid& grid = comm->get_model_grid();
  if (comm->am_world_master()) {
    const string train_data = trainParams.DatasetRootDir + trainParams.TrainFile;
    const string test_data  = trainParams.DatasetRootDir + trainParams.TestFile;

    cout << "Number of models: " << comm->get_num_models() << endl;
    cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
    cout << "Train Data: " << train_data << endl;
    cout << "Test Data: " << test_data << endl;
    cout << "EpochCount: " << trainParams.EpochCount << endl;
    cout << "MBSize: " << trainParams.MBSize << endl;
    cout << "LearnRateMethod: " << trainParams.LearnRateMethod << endl;
    cout << "LearnRate: " << trainParams.LearnRate << endl;
    cout << "LrDecayRate: " << trainParams.LrDecayRate << endl;
    cout << "LrDecayCycles: " << trainParams.LrDecayCycles << endl;
    cout << "LrMomentum: " << trainParams.LrMomentum << endl;
    cout << "ActivationType: " << static_cast<int>(trainParams.ActivationType) << endl;
    cout << "DropOut: " << trainParams.DropOut << endl;
    cout << "InitialWeight: " << static_cast<int>(trainParams.WeightInitType) << endl;
    cout << "ProcsPerModel: " << trainParams.ProcsPerModel << endl;
    cout << "PercentageTrainingSamples: " << trainParams.PercentageTrainingSamples << endl;
    cout << "PercentageValidationSamples: " << trainParams.PercentageValidationSamples << endl;
    cout << "BlockSize: " << perfParams.BlockSize << endl;
    cout << endl;
  }
}

//@todo use param options

int main(int argc, char* argv[])
{
    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);
    init_random(42);
    init_data_seq_random(42);

    try {


        TrainingParams trainParams;
        PerformanceParams perfParams;

        // Read in the user specified network topology
        NetworkParams netParams;

        init_params(trainParams, perfParams);

        const string train_data = trainParams.DatasetRootDir + trainParams.TrainFile;
        const string test_data  = trainParams.DatasetRootDir + trainParams.TestFile;

        // Set up the communicator and get the grid.
        lbann_comm* comm = new lbann_comm(trainParams.ProcsPerModel);
        Grid& grid = comm->get_model_grid();

        print_params(comm, trainParams, perfParams);

        int parallel_io = perfParams.MaxParIOSize;
        if (parallel_io == 0) {
          if (comm->am_world_master()) {
            cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
          }
          parallel_io = grid.Size();
        } else {
          if (comm->am_world_master()) {
            cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
          }
        }

        ///////////////////////////////////////////////////////////////////
        // load training data
        ///////////////////////////////////////////////////////////////////
        clock_t load_time = clock();
        data_reader_nci_regression nci_trainset(trainParams.MBSize, true);
        nci_trainset.set_data_filename(train_data);
        nci_trainset.set_use_percent(trainParams.PercentageTrainingSamples);
        nci_trainset.load();

        if (comm->am_world_master()) {
          cout << "Training using " << (trainParams.PercentageTrainingSamples*100) << "% of the training data set, which is " << nci_trainset.getNumData() << " samples." << endl;
        }

        ///////////////////////////////////////////////////////////////////
        // create a validation set from the unused training data (NCI)
        ///////////////////////////////////////////////////////////////////
        data_reader_nci_regression nci_validation_set(nci_trainset); // Clone the training set object
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
        // load testing data (NCI)
        ///////////////////////////////////////////////////////////////////
        data_reader_nci_regression nci_testset(trainParams.MBSize, true);
        nci_testset.set_data_filename(test_data);
        nci_testset.load();
       
        if (comm->am_world_master()) {
          cout << "Load Time " << ((double)clock() - load_time) / CLOCKS_PER_SEC << endl;
        }

        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////

        // Initialize optimizer
        Optimizer_factory *optimizer; //@todo replace with factory
        if (trainParams.LearnRateMethod == 1) { // Adagrad
          optimizer = new Adagrad_factory(comm, trainParams.LearnRate);
        }else if (trainParams.LearnRateMethod == 2) { // RMSprop
          optimizer = new RMSprop_factory(comm, trainParams.LearnRate);
        }else if (trainParams.LearnRateMethod == 3) { // Adam
          optimizer = new Adam_factory(comm, trainParams.LearnRate);
        }else if (trainParams.LearnRateMethod == 4) { // SGD
          optimizer = new SGD_factory(comm, trainParams.LearnRate, trainParams.LrMomentum, trainParams.LrDecayRate, true);
        }else {
          cout << "undefined learning rate method" << endl;
          return -1;
        }

        // Initialize network
        layer_factory* lfac = new layer_factory();
        deep_neural_network dnn(trainParams.MBSize, comm, new objective_functions::mean_squared_error(comm), lfac, optimizer);

        metrics::mean_squared_error mse(comm);
        dnn.add_metric(&mse);

        std::map<execution_mode, DataReader*> data_readers = {std::make_pair(execution_mode::training,&nci_trainset),
                                                              std::make_pair(execution_mode::validation, &nci_validation_set),
                                                              std::make_pair(execution_mode::testing, &nci_testset)};

        input_layer *ilayer = new input_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io,
                                  (int) trainParams.MBSize, data_readers);
        dnn.add(ilayer);

        dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, 4096, trainParams.ActivationType, trainParams.WeightInitType, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
        dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, 1024, trainParams.ActivationType, trainParams.WeightInitType, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
        dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, 256, trainParams.ActivationType, trainParams.WeightInitType, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
        dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, 64, trainParams.ActivationType, trainParams.WeightInitType, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
        dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, 16, trainParams.ActivationType, trainParams.WeightInitType, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
        dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, 1, activation_type::ID, trainParams.WeightInitType, {});


        target_layer *tlayer = new target_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io,
                                   (int) trainParams.MBSize, data_readers, true, true);
        if (! tlayer->is_for_regression()) {
            if (comm->am_world_master()) cout << "Target layer is not set for regression" << endl;
            return 0;
        }
        dnn.add(tlayer);

        //lbann_summary summarizer("/p/lscratchf/jacobs32", comm);
        // Print out information for each epoch.
        lbann_callback_print print_cb;
        dnn.add_callback(&print_cb);
        // Record training time information.
        //lbann_callback_timer timer_cb(&summarizer);
        //dnn.add_callback(&timer_cb);
        // Summarize information to Tensorboard.
        //lbann_callback_summary summary_cb(&summarizer, 25);
        //dnn.add_callback(&summary_cb);
        lbann_callback_early_stopping stopping_cb(1);
        dnn.add_callback(&stopping_cb);

        if (comm->am_world_master()) {
          cout << "Layer initialized:" << endl;
          cout << "Print Layers using factory ";
          lfac->print();
          cout << endl;
        }

        if (comm->am_world_master()) {
          cout << "Parameter settings:" << endl;
          cout << "\tMini-batch size: " << trainParams.MBSize << endl;
          cout << "\tLearning rate: " << trainParams.LearnRate << endl << endl;
          cout << "\tEpoch count: " << trainParams.EpochCount << endl;
        }

        ///////////////////////////////////////////////////////////////////
        // main loop for training/testing
        ///////////////////////////////////////////////////////////////////

        // Initialize the model's data structures
        dnn.setup();

        // train/test
        for (int t = 0; t < trainParams.EpochCount; t++) {
          if (comm->am_world_master()) {
            cout << "training " << t << "th round ..." << endl;
          }
          dnn.train(1, true);
          if (comm->am_world_master()) {
            cout << "testing " << t << "th round ..." << endl;
          }
          // testing
          dnn.evaluate(execution_mode::testing);
        }
        if (comm->am_world_master()) {
          cout << "completing..." << endl;
        }
        delete optimizer;
        delete comm;
    }
    catch (exception& e) { ReportException(e); }

    cout << "Finalizing..." << endl;
    // free all resources by El and MPI
    Finalize();

    cout << "Exiting..." << endl;
    return 0;
}
