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
// lbann_dnn_mnist.cpp - DNN application for mnist
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_mnist.hpp"
#include "lbann/lbann.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

using namespace std;
using namespace lbann;
using namespace El;


// layer definition
const std::vector<int> g_LayerDim = {784, 100, 30, 10};
const uint g_NumLayers = g_LayerDim.size(); // # layers

/// Main function
int main(int argc, char* argv[])
{
    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);
    init_random(42);
    lbann_comm* comm = NULL;

    try {

        // Get data files
        const string g_MNIST_TrainLabelFile = Input("--train-label-file",
                                                    "MNIST training set label file",
                                                    "train-labels-idx1-ubyte");
        const string g_MNIST_TrainImageFile = Input("--train-image-file",
                                                    "MNIST training set image file",
                                                    "train-images-idx3-ubyte");
        const string g_MNIST_TestLabelFile = Input("--test-label-file",
                                                   "MNIST test set label file",
                                                   "t10k-labels-idx1-ubyte");
        const string g_MNIST_TestImageFile = Input("--test-image-file",
                                                   "MNIST test set image file",
                                                   "t10k-images-idx3-ubyte");

        ///////////////////////////////////////////////////////////////////
        // initalize grid, block
        ///////////////////////////////////////////////////////////////////

        // Initialize parameter defaults
        TrainingParams trainParams;
        trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/MNIST/";
        trainParams.EpochCount = 20;
        trainParams.MBSize = 256;
        trainParams.LearnRate = 0.01;
        trainParams.DropOut = 0.5;
        trainParams.SummaryDir = "./out";
        trainParams.ProcsPerModel = 0;
        trainParams.PercentageTrainingSamples = 0.90;
        trainParams.PercentageValidationSamples = 1.00;
        PerformanceParams perfParams;
        perfParams.BlockSize = 256;

        // Parse command-line inputs
        trainParams.parse_params();
        perfParams.parse_params();

        bool scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
        bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", false);
        bool unit_variance = Input("--unit-variance", "standardize to unit-variance", false);

        //if set to true, above three settings have no effect
        bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

        ProcessInput();
        PrintInputReport();

        // Set algorithmic blocksize
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
        DataReader_MNIST mnist_trainset(trainParams.MBSize, true);
        if (!mnist_trainset.load(trainParams.DatasetRootDir,
                                 g_MNIST_TrainImageFile,
                                 g_MNIST_TrainLabelFile, trainParams.PercentageTrainingSamples)) {
          if (comm->am_world_master()) {
            cout << "MNIST train data error" << endl;
          }
          return -1;
        }
        if (comm->am_world_master()) {
          cout << "Training using " << (trainParams.PercentageTrainingSamples*100) << "% of the training data set, which is " << mnist_trainset.getNumData() << " samples." << endl;
        }

        mnist_trainset.scale(scale);
        mnist_trainset.subtract_mean(subtract_mean);
        mnist_trainset.unit_variance(unit_variance);
        mnist_trainset.z_score(z_score);

        ///////////////////////////////////////////////////////////////////
        // create a validation set from the unused training data (MNIST)
        ///////////////////////////////////////////////////////////////////
        DataReader_MNIST mnist_validation_set(mnist_trainset); // Clone the training set object
        if (!mnist_validation_set.swap_used_and_unused_index_sets()) { // Swap the used and unused index sets so that it validates on the remaining data
          if (comm->am_world_master()) {
            cout << "MNIST validation data error" << endl;
          }
          return -1;
        }

        if(trainParams.PercentageValidationSamples == 1.00) {
          if (comm->am_world_master()) {
            cout << "Validating training using " << ((1.00 - trainParams.PercentageTrainingSamples)*100) << "% of the training data set, which is " << mnist_validation_set.getNumData() << " samples." << endl;
          }
        }else {
          size_t preliminary_validation_set_size = mnist_validation_set.getNumData();
          size_t final_validation_set_size = mnist_validation_set.trim_data_set(trainParams.PercentageValidationSamples);
          if (comm->am_world_master()) {
            cout << "Trim the validation data set from " << preliminary_validation_set_size << " samples to " << final_validation_set_size << " samples." << endl;
          }
        }

        ///////////////////////////////////////////////////////////////////
        // load testing data (MNIST)
        ///////////////////////////////////////////////////////////////////
        DataReader_MNIST mnist_testset(trainParams.MBSize, true);
        if (!mnist_testset.load(trainParams.DatasetRootDir,
                                g_MNIST_TestImageFile,
                                g_MNIST_TestLabelFile,
                                trainParams.PercentageTestingSamples)) {
          if (comm->am_world_master()) {
            cout << "MNIST Test data error" << endl;
          }
          return -1;
        }
        if (comm->am_world_master()) {
          cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << mnist_testset.getNumData() << " samples." << endl;
        }

        mnist_testset.scale(scale);
        mnist_testset.subtract_mean(subtract_mean);
        mnist_testset.unit_variance(unit_variance);
        mnist_trainset.z_score(z_score);

        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////

        // Initialize optimizer
        Optimizer_factory *optimizer;
        if (trainParams.LearnRateMethod == 1) { // Adagrad
          optimizer = new Adagrad_factory(comm, trainParams.LearnRate);
        }else if (trainParams.LearnRateMethod == 2) { // RMSprop
          optimizer = new RMSprop_factory(comm, trainParams.LearnRate);
        }else {
          optimizer = new SGD_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
        }

        // Initialize network
        layer_factory* lfac = new layer_factory();
#if __LIB_CUDNN
        cudnn::cudnn_manager* cudnn = new cudnn::cudnn_manager(comm);
#else // __LIB_CUDNN
        cudnn::cudnn_manager* cudnn = NULL;
#endif // __LIB_CUDNN
        deep_neural_network dnn(trainParams.MBSize, comm, new categorical_cross_entropy(comm), lfac, optimizer);
        std::map<execution_mode, DataReader*> data_readers = {std::make_pair(execution_mode::training,&mnist_trainset), 
                                                               std::make_pair(execution_mode::validation, &mnist_validation_set), 
                                                               std::make_pair(execution_mode::testing, &mnist_testset)};
        //input_layer *input_layer = new input_layer_distributed_minibatch(comm,  (int) trainParams.MBSize, data_readers);
        input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, data_readers);
        dnn.add(input_layer);

        // First convolution layer
        {
          Optimizer* convolution_layer_optimizer = optimizer->create_optimizer(matrix_format::STAR_STAR);
          int numDims = 2;
          int inputChannels = 1;
          int inputDims[] = {28, 28};
          int outputChannels = 32;
          int filterDims[] = {3, 3};
          int convPads[] = {0, 0};
          int convStrides[] = {1, 1};
          convolutional_layer* layer
            = new convolutional_layer(1, numDims, inputChannels, inputDims,
                                      outputChannels, filterDims,
                                      convPads, convStrides,
                                      trainParams.MBSize,
                                      activation_type::RELU,
                                      weight_initialization::glorot_uniform,
                                      comm, convolution_layer_optimizer, 
                                      {}, cudnn);
          dnn.add(layer);
        }

        // Second convolution layer
        {
          Optimizer* convolution_layer_optimizer = optimizer->create_optimizer(matrix_format::STAR_STAR);
          int numDims = 2;
          int inputChannels = 32;
          int inputDims[] = {26, 26};
          int outputChannels = 32;
          int filterDims[] = {3, 3};
          int convPads[] = {0, 0};
          int convStrides[] = {1, 1};
          convolutional_layer* layer
            = new convolutional_layer(2, numDims, inputChannels, inputDims,
                                      outputChannels, filterDims,
                                      convPads, convStrides,
                                      trainParams.MBSize,
                                      activation_type::RELU,
                                      weight_initialization::glorot_uniform,
                                      comm, convolution_layer_optimizer,
                                      {},
                                      cudnn);
          dnn.add(layer);
        }

        // Pooling layer
        {
          int numDims = 2;
          int channels = 32;
          int inputDim[] = {24, 24};
          int poolWindowDims[] = {2, 2};
          int poolPads[] = {0, 0};
          int poolStrides[] = {2, 2};
          pool_mode poolMode = pool_mode::max;
          pooling_layer* layer
            = new pooling_layer(3, numDims, channels, inputDim,
                                poolWindowDims, poolPads, poolStrides, poolMode,
                                trainParams.MBSize, activation_type::ID,
                                comm,
                                {new dropout(comm, 0.75)},
                                cudnn);
          dnn.add(layer);
        }

        // Fully connected and output layers
        dnn.add("FullyConnected", 128, trainParams.ActivationType,
                weight_initialization::glorot_uniform, {new dropout(comm, 0.5)});
        dnn.add("Softmax", 10, activation_type::ID,
                weight_initialization::glorot_uniform, {});

        //target_layer *target_layer = new target_layer_distributed_minibatch(comm, (int) trainParams.MBSize, data_readers, true);
        target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, data_readers, true);
        dnn.add(target_layer);

        lbann_summary summarizer(trainParams.SummaryDir, comm);
        lbann_callback_print print_cb;
        dnn.add_callback(&print_cb);
        // lbann_callback_io io_cb({0,3});
        // dnn.add_callback(&io_cb);

        // Summarize information to Tensorboard
        lbann_callback_summary summary_cb(&summarizer, 25);
        dnn.add_callback(&summary_cb);

        // Initialize the model's data structures
        dnn.setup();
        if (comm->am_world_master()) {
          cout << "Layer initialized:" << endl;
          for (uint n = 0; n < g_NumLayers; n++) {
            cout << "\tLayer[" << n << "]: " << g_LayerDim[n] << endl;
          }
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

        // train/test
        for (int t = 0; t < trainParams.EpochCount; t++) {
            dnn.train(1, true);
            dnn.evaluate(execution_mode::testing);
        }

        // Free dynamically allocated memory
        // delete lfac;  // Causes segfault
        delete optimizer;
        // delete comm;  // Causes error

    }
    catch (lbann_exception& e) { lbann_report_exception(e, comm); }
    catch (exception& e) { ReportException(e); } /// Elemental exceptions

    // free all resources by El and MPI
    Finalize();

    return 0;
}
