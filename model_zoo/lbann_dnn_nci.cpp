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
#include "lbann/data_readers/lbann_data_reader_nci.hpp"
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

  try {


        ///////////////////////////////////////////////////////////////////
        // initalize grid, block
        ///////////////////////////////////////////////////////////////////
      TrainingParams trainParams;
      trainParams.DatasetRootDir = "/usr/mic/post1/metagenomics/cancer/anl_datasets/tmp_norm/";
      trainParams.EpochCount = 10;
      trainParams.MBSize = 50;
      trainParams.LearnRate = 0.0001;
      trainParams.DropOut = -1.0f;
      trainParams.ProcsPerModel = 0;
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
      data_reader_nci nci_dataset(trainParams.MBSize, true);
      if (!nci_dataset.load(train_data)) {
        if (comm->am_world_master()) {
          cout << "NCI train data error" << endl;
        }
        return -1;
      }


        ///////////////////////////////////////////////////////////////////
        // load testing data (MNIST)
        ///////////////////////////////////////////////////////////////////
      data_reader_nci nci_testset(trainParams.MBSize, true);
      if (!nci_testset.load(test_data)) {
        if (comm->am_world_master()) {
          cout << "NCI Test data error" << endl;
        }
        return -1;
      }

      if (comm->am_world_master()) cout << "Load Time " << ((double)clock() - load_time) / CLOCKS_PER_SEC << endl;

        //nci_testset.setup(-1);



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
      Dnn dnn(trainParams.MBSize,
              trainParams.Lambda,
              optimizer, comm, lfac);

      input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(comm, parallel_io,
                                (int) trainParams.MBSize, &nci_dataset, &nci_testset);
      dnn.add(input_layer);

      dnn.add("FullyConnected", 500, trainParams.ActivationType, {new dropout(trainParams.DropOut)});
      dnn.add("FullyConnected", 300, trainParams.ActivationType, {new dropout(trainParams.DropOut)});
      dnn.add("SoftMax", 2);


      target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(comm, parallel_io,
                                 (int) trainParams.MBSize, &nci_dataset, &nci_testset, true);
      dnn.add(target_layer);

      lbann_summary summarizer("/p/lscratchf/jacobs32", comm);
      // Print out information for each epoch.
      lbann_callback_print print_cb;
      dnn.add_callback(&print_cb);
      // Record training time information.
      lbann_callback_timer timer_cb(&summarizer);
      dnn.add_callback(&timer_cb);
      // Summarize information to Tensorboard.
      lbann_callback_summary summary_cb(&summarizer, 25);
      dnn.add_callback(&summary_cb);

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

         //train/test
      for(int t = 0; t < trainParams.EpochCount; t++) {
        dnn.train(1);
        // testing
        DataType accuracy = dnn.evaluate();
      }
      delete optimizer;
      delete comm;
    }
    catch (exception& e) { ReportException(e); }

    // free all resources by El and MPI
    Finalize();

    return 0;
}
