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
// lbann_dnn_imagenet.cpp - DNN application for image-net classification
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/regularization/lbann_dropout.hpp"

#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <iomanip>
#include <string>

//#include <algorithm>
//#include <random>

using namespace std;
using namespace lbann;
using namespace El;


// train/test data info
const string g_ImageNet_TrainDir = "resized_256x256/train/";
const string g_ImageNet_ValDir = "resized_256x256/val/";
const string g_ImageNet_TestDir = "resized_256x256/val/"; //test/";
const string g_ImageNet_LabelDir = "labels/";
const string g_ImageNet_TrainLabelFile = "train_c0-9.txt";
const string g_ImageNet_ValLabelFile = "val.txt";
const string g_ImageNet_TestLabelFile = "val_c0-9.txt"; //"test.txt";



int main(int argc, char* argv[])
{
    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);
    lbann_comm *comm = NULL;

    try {
        ///////////////////////////////////////////////////////////////////
        // initalize grid, block
        ///////////////////////////////////////////////////////////////////
        TrainingParams trainParams;
        trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/ILSVRC2012/";
        trainParams.DropOut = 0.1;
        trainParams.ProcsPerModel = 0;
        trainParams.parse_params();
        trainParams.PercentageTrainingSamples = 0.80;
        trainParams.PercentageValidationSamples = 1.00;
        PerformanceParams perfParams;
        perfParams.parse_params();
        // Read in the user specified network topology
        NetworkParams netParams;
        netParams.parse_params();
        // Get some environment variables from the launch
        SystemParams sysParams;
        sysParams.parse_params();


        int decayIterations = 1;

        ProcessInput();
        PrintInputReport();

        // set algorithmic blocksize
        SetBlocksize(perfParams.BlockSize);


        // Set up the communicator and get the grid.
        comm = new lbann_comm(trainParams.ProcsPerModel);
        Grid& grid = comm->get_model_grid();
        if (comm->am_world_master()) {
          cout << "Number of models: " << comm->get_num_models() << endl;
          cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
          cout << endl;
        }

        int parallel_io = perfParams.MaxParIOSize;
        if(parallel_io == 0) {
          if(comm->am_world_master()) {
             cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() << " (Limited to # Processes)" << endl;
          }
          parallel_io = comm->get_procs_per_model();
        }else {
          if(comm->am_world_master()) {
            cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
          }
        }

        parallel_io = 1;
        ///////////////////////////////////////////////////////////////////
        // load training data (ImageNet)
        ///////////////////////////////////////////////////////////////////
        DataReader_ImageNet imagenet_trainset(trainParams.MBSize, true);
        bool training_set_loaded = false;
        training_set_loaded = imagenet_trainset.load(trainParams.DatasetRootDir + g_ImageNet_TrainDir,
                                                     trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile,
                                                     trainParams.PercentageTrainingSamples);
        if (!training_set_loaded) {
          if (comm->am_world_master()) {
            cout << "ImageNet train data error" << endl;
          }
          return -1;
        }
        if (comm->am_world_master()) {
          cout << "Training using " << (trainParams.PercentageTrainingSamples*100) << "% of the training data set, which is " << imagenet_trainset.getNumData() << " samples." << endl;
        }

        ///////////////////////////////////////////////////////////////////
        // create a validation set from the unused training data (ImageNet)
        ///////////////////////////////////////////////////////////////////
        DataReader_ImageNet imagenet_validation_set(imagenet_trainset); // Clone the training set object
        if (!imagenet_validation_set.swap_used_and_unused_index_sets()) { // Swap the used and unused index sets so that it validates on the remaining data
          if (comm->am_world_master()) {
            cout << "ImageNet validation data error" << endl;
          }
          return -1;
        }

        if(trainParams.PercentageValidationSamples == 1.00) {
          if (comm->am_world_master()) {
            cout << "Validating training using " << ((1.00 - trainParams.PercentageTrainingSamples)*100) << "% of the training data set, which is " << imagenet_validation_set.getNumData() << " samples." << endl;
          }
        }else {
          size_t preliminary_validation_set_size = imagenet_validation_set.getNumData();
          size_t final_validation_set_size = imagenet_validation_set.trim_data_set(trainParams.PercentageValidationSamples);
          if (comm->am_world_master()) {
            cout << "Trim the validation data set from " << preliminary_validation_set_size << " samples to " << final_validation_set_size << " samples." << endl;
          }
        }

        ///////////////////////////////////////////////////////////////////
        // load testing data (ImageNet)
        ///////////////////////////////////////////////////////////////////
        DataReader_ImageNet imagenet_testset(trainParams.MBSize, true);
        bool testing_set_loaded = false;
        testing_set_loaded = imagenet_testset.load(trainParams.DatasetRootDir + g_ImageNet_TestDir,
                                                   trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile,
                                                   trainParams.PercentageTestingSamples);
        if (!testing_set_loaded) {
          if (comm->am_world_master()) {
            cout << "ImageNet Test data error" << endl;
          }
          return -1;
        }
        if (comm->am_world_master()) {
          cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << imagenet_testset.getNumData() << " samples." << endl;
        }

        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////
        Optimizer_factory *optimizer;
        if (trainParams.LearnRateMethod == 1) { // Adagrad
          optimizer = new Adagrad_factory(comm, trainParams.LearnRate);
        }else if (trainParams.LearnRateMethod == 2) { // RMSprop
          optimizer = new RMSprop_factory(comm/*, trainParams.LearnRate*/);
        }else {
          optimizer = new SGD_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
        }

        layer_factory* lfac = new layer_factory();
        greedy_layerwise_autoencoder* gla = new greedy_layerwise_autoencoder(trainParams.MBSize, comm, lfac, optimizer);
        std::map<execution_mode, DataReader*> data_readers = {std::make_pair(execution_mode::training,&imagenet_trainset),
                                                              std::make_pair(execution_mode::validation, &imagenet_validation_set),
                                                              std::make_pair(execution_mode::testing, &imagenet_testset)};
        input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, data_readers);
        gla->add(input_layer);
        gla->add("FullyConnected", 10000, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(comm, trainParams.DropOut)});
        gla->add("FullyConnected", 5000, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(comm,trainParams.DropOut)});
        gla->add("FullyConnected", 2000, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(comm,trainParams.DropOut)});
        gla->add("FullyConnected", 1000, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(comm,trainParams.DropOut)});
        gla->add("FullyConnected", 500, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(comm,trainParams.DropOut)});

        lbann_summary summarizer("/p/lscratchf/jacobs32", comm);
        // Print out information for each epoch.
        lbann_callback_print print_cb;
        gla->add_callback(&print_cb);
        // Record training time information.
        lbann_callback_timer timer_cb(&summarizer);
        gla->add_callback(&timer_cb);
        // Summarize information to Tensorboard.
        lbann_callback_summary summary_cb(&summarizer, 25);
        gla->add_callback(&summary_cb);
        // lbann_callback_io io_cb({0});
        // dnn->add_callback(&io_cb);

        gla->setup();

        if (comm->am_world_master()) {
	        cout << "Layer initialized:" << endl;
                for (uint n = 0; n < gla->get_layers().size(); n++)
                  cout << "\tLayer[" << n << "]: " << gla->get_layers()[n]->NumNeurons << endl;
            cout << endl;

	    cout << "Parameter settings:" << endl;
            cout << "\tBlock size: " << perfParams.BlockSize << endl;
            cout << "\tEpochs: " << trainParams.EpochCount << endl;
            cout << "\tMini-batch size: " << trainParams.MBSize << endl;
            cout << "\tLearning rate: " << trainParams.LearnRate << endl;
            cout << "\tEpoch count: " << trainParams.EpochCount << endl << endl;
            if(perfParams.MaxParIOSize == 0) {
              cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
            }else {
              cout << "\tMax Parallel I/O Fetch: " << perfParams.MaxParIOSize << endl;
            }
            cout << "\tDataset: " << trainParams.DatasetRootDir << endl;
        }




        mpi::Barrier(grid.Comm());

        gla->train(trainParams.EpochCount);

        delete gla;
    }
    catch (lbann_exception& e) { lbann_report_exception(e, comm); }
    catch (exception& e) { ReportException(e); } /// Elemental exceptions

    // free all resources by El and MPI
    Finalize();

    return 0;
}
