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
// lbann_alexnet.cpp - AlexNet application for ImageNet classification
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/regularization/lbann_dropout.hpp"
#include "lbann/data_readers/lbann_image_utils.hpp"

#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <iomanip>
#include <string>

using namespace std;
using namespace lbann;
using namespace El;



// train/test data info
const int g_SaveImageIndex[1] = {0}; // for auto encoder
//const int g_SaveImageIndex[5] = {293, 2138, 3014, 6697, 9111}; // for auto encoder
//const int g_SaveImageIndex[5] = {1000, 2000, 3000, 4000, 5000}; // for auto encoder
const string g_ImageNet_TrainDir = "resized_256x256/train/";
const string g_ImageNet_ValDir = "resized_256x256/val/";
const string g_ImageNet_TestDir = "resized_256x256/test/";
const string g_ImageNet_LabelDir = "labels/";
const string g_ImageNet_TrainLabelFile = "train.txt"; // "train_c0-9.txt";
const string g_ImageNet_ValLabelFile = "val.txt";
const string g_ImageNet_TestLabelFile = "test.txt"; //"val_c0-9.txt"; //"test.txt";
const uint g_ImageNet_Width = 256;
const uint g_ImageNet_Height = 256;

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

        // training settings
        int decayIterations = 1;

        bool scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
        bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", false);
        bool unit_variance = Input("--unit-variance", "standardize to unit-variance", false);

        //if set to true, above three settings have no effect
        bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

        ProcessInput();
        PrintInputReport();

        // set algorithmic blocksize
        SetBlocksize(perfParams.BlockSize);

        // create timer for performance measurement
        Timer timer_io;
        Timer timer_lbann;
        Timer timer_val;
        double sec_all_io = 0;
        double sec_all_lbann = 0;
        double sec_all_val = 0;

        // Set up the communicator and get the grid.
        comm = new lbann_comm(trainParams.ProcsPerModel);
        Grid& grid = comm->get_model_grid();
        if (comm->am_world_master()) {
          cout << "Number of models: " << comm->get_num_models() << endl;
          cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
          cout << endl;
        }

        int parallel_io = perfParams.MaxParIOSize;
        //        int io_offset = 0;
        if(parallel_io == 0) {
          if(comm->am_world_master()) {
             cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() << " (Limited to # Processes)" << endl;
          }
          parallel_io = comm->get_procs_per_model();
          //          io_offset = comm->get_rank_in_model() *trainParams.MBSize;
        }else {
          if(comm->am_world_master()) {
            cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
          }
          //          parallel_io = grid.Size();
          // if(perfParams.MaxParIOSize > 1) {
          //   io_offset = comm->get_rank_in_model() *trainParams.MBSize;
          // }
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
            cerr << __FILE__ << " " << __LINE__ << " ImageNet train data error" << endl;
          }
          return -1;
        }
        if (comm->am_world_master()) {
          cout << "Training using " << (trainParams.PercentageTrainingSamples*100) << "% of the training data set, which is " << imagenet_trainset.getNumData() << " samples." << endl;
        }

        imagenet_trainset.scale(scale);
        imagenet_trainset.subtract_mean(subtract_mean);
        imagenet_trainset.unit_variance(unit_variance);
        imagenet_trainset.z_score(z_score);

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
            cerr << __FILE__ << " " << __LINE__ << " ImageNet Test data error" << endl;
          }
          return -1;
        }
        if (comm->am_world_master()) {
          cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << imagenet_testset.getNumData() << " samples." << endl;
        }
        imagenet_testset.scale(scale);
        imagenet_testset.subtract_mean(subtract_mean);
        imagenet_testset.unit_variance(unit_variance);
        imagenet_testset.z_score(z_score);

        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////

        // Initialize optimizer factory
        Optimizer_factory *optimizer;
        if (trainParams.LearnRateMethod == 1) { // Adagrad
          optimizer = new Adagrad_factory(comm, trainParams.LearnRate);
        }else if (trainParams.LearnRateMethod == 2) { // RMSprop
          optimizer = new RMSprop_factory(comm/*, trainParams.LearnRate*/);
        }else {
          optimizer = new SGD_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
        }

        // Initialize layer factory
        layer_factory* lfac = new layer_factory();

        // Initialize cuDNN (if detected)
#if __LIB_CUDNN
        cudnn::cudnn_manager* cudnn = new cudnn::cudnn_manager(comm);
#else // __LIB_CUDNN
        cudnn::cudnn_manager* cudnn = NULL;
#endif // __LIB_CUDNN

        deep_neural_network *dnn = NULL;
        dnn = new deep_neural_network(trainParams.MBSize, comm, new objective_functions::categorical_cross_entropy(comm), lfac, optimizer);
        std::map<execution_mode, DataReader*> data_readers = {std::make_pair(execution_mode::training,&imagenet_trainset), 
                                                              std::make_pair(execution_mode::validation, &imagenet_validation_set), 
                                                              std::make_pair(execution_mode::testing, &imagenet_testset)};
        input_layer *input_layer = new input_layer_distributed_minibatch(comm, (int) trainParams.MBSize, data_readers);
        // input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, data_readers);
        dnn->add(input_layer);

        // Layer 1 (convolutional)
        {
          Optimizer* convolution_layer_optimizer = optimizer->create_optimizer(matrix_format::STAR_STAR);
          int numDims = 2;
          int inputChannels = 1; // TODO: this should be 3
          int inputDims[] = {256, 256};
          int outputChannels = 96;
          int filterDims[] = {11, 11};
          int convPads[] = {2, 2};
          int convStrides[] = {4, 4};
          convolutional_layer* layer
            = new convolutional_layer(1, numDims, inputChannels, inputDims,
                                      outputChannels, filterDims,
                                      convPads, convStrides,
                                      trainParams.MBSize,
                                      activation_type::RELU,
                                      weight_initialization::glorot_uniform,
                                      comm, convolution_layer_optimizer, 
                                      {}, cudnn);
          dnn->add(layer);
        }

        // Layer 2 (pooling)
        {
          int numDims = 2;
          int channels = 96;
          int inputDim[] = {63, 63};
          int poolWindowDims[] = {3, 3};
          int poolPads[] = {0, 0};
          int poolStrides[] = {2, 2};
          pool_mode poolMode = pool_mode::max;
          pooling_layer* layer
            = new pooling_layer(2, numDims, channels, inputDim,
                                poolWindowDims, poolPads, poolStrides, poolMode,
                                trainParams.MBSize, activation_type::ID,
                                comm,
                                {},
                                cudnn);
          dnn->add(layer);
        }

        // Layer 3 (convolutional)
        {
          Optimizer* convolution_layer_optimizer = optimizer->create_optimizer(matrix_format::STAR_STAR);
          int numDims = 2;
          int inputChannels = 96;
          int inputDims[] = {31, 31};
          int outputChannels = 256;
          int filterDims[] = {5, 5};
          int convPads[] = {0, 0};
          int convStrides[] = {1, 1};
          convolutional_layer* layer
            = new convolutional_layer(3, numDims, inputChannels, inputDims,
                                      outputChannels, filterDims,
                                      convPads, convStrides,
                                      trainParams.MBSize,
                                      activation_type::RELU,
                                      weight_initialization::glorot_uniform,
                                      comm, convolution_layer_optimizer, 
                                      {}, cudnn);
          dnn->add(layer);
        }

        // Layer 4 (pooling)
        {
          int numDims = 2;
          int channels = 256;
          int inputDim[] = {27, 27};
          int poolWindowDims[] = {3, 3};
          int poolPads[] = {0, 0};
          int poolStrides[] = {2, 2};
          pool_mode poolMode = pool_mode::max;
          pooling_layer* layer
            = new pooling_layer(4, numDims, channels, inputDim,
                                poolWindowDims, poolPads, poolStrides, poolMode,
                                trainParams.MBSize, activation_type::ID,
                                comm,
                                {},
                                cudnn);
          dnn->add(layer);
        }

        // Layer 5 (convolutional)
        {
          Optimizer* convolution_layer_optimizer = optimizer->create_optimizer(matrix_format::STAR_STAR);
          int numDims = 2;
          int inputChannels = 256;
          int inputDims[] = {13, 13};
          int outputChannels = 384;
          int filterDims[] = {3, 3};
          int convPads[] = {1, 1};
          int convStrides[] = {1, 1};
          convolutional_layer* layer
            = new convolutional_layer(5, numDims, inputChannels, inputDims,
                                      outputChannels, filterDims,
                                      convPads, convStrides,
                                      trainParams.MBSize,
                                      activation_type::RELU,
                                      weight_initialization::glorot_uniform,
                                      comm, convolution_layer_optimizer, 
                                      {}, cudnn);
          dnn->add(layer);
        }

        // Layer 6 (convolutional)
        {
          Optimizer* convolution_layer_optimizer = optimizer->create_optimizer(matrix_format::STAR_STAR);
          int numDims = 2;
          int inputChannels = 384;
          int inputDims[] = {13, 13};
          int outputChannels = 384;
          int filterDims[] = {3, 3};
          int convPads[] = {1, 1};
          int convStrides[] = {1, 1};
          convolutional_layer* layer
            = new convolutional_layer(6, numDims, inputChannels, inputDims,
                                      outputChannels, filterDims,
                                      convPads, convStrides,
                                      trainParams.MBSize,
                                      activation_type::RELU,
                                      weight_initialization::glorot_uniform,
                                      comm, convolution_layer_optimizer, 
                                      {}, cudnn);
          dnn->add(layer);
        }

        // Layer 7 (convolutional)
        {
          Optimizer* convolution_layer_optimizer = optimizer->create_optimizer(matrix_format::STAR_STAR);
          int numDims = 2;
          int inputChannels = 384;
          int inputDims[] = {13, 13};
          int outputChannels = 256;
          int filterDims[] = {3, 3};
          int convPads[] = {1, 1};
          int convStrides[] = {1, 1};
          convolutional_layer* layer
            = new convolutional_layer(7, numDims, inputChannels, inputDims,
                                      outputChannels, filterDims,
                                      convPads, convStrides,
                                      trainParams.MBSize,
                                      activation_type::RELU,
                                      weight_initialization::glorot_uniform,
                                      comm, convolution_layer_optimizer, 
                                      {}, cudnn);
          dnn->add(layer);
        }

        // Layer 8 (fully-connected)
        dnn->add("FullyConnected",
                 4096,
                 activation_type::RELU,
                 weight_initialization::glorot_uniform,
                 {new dropout(comm, 0.5)});

        // Layer 9 (fully-connected)
        dnn->add("FullyConnected",
                 4096,
                 activation_type::RELU,
                 weight_initialization::glorot_uniform,
                 {new dropout(comm, 0.5)});

        // Layer 10 (softmax)
        dnn->add("Softmax",
                 1000,
                 activation_type::ID,
                 weight_initialization::glorot_uniform,
                 {});

        target_layer *target_layer = new target_layer_distributed_minibatch(comm, (int) trainParams.MBSize, data_readers, true);
        // target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int) trainParams.MBSize, data_readers, true);
        dnn->add(target_layer);

        lbann_summary summarizer("/p/lscratchf/vanessen", comm);
        // Print out information for each epoch.
        lbann_callback_print print_cb;
        dnn->add_callback(&print_cb);
        // Record training time information.
        lbann_callback_timer timer_cb(&summarizer);
        dnn->add_callback(&timer_cb);
        // Summarize information to Tensorboard.
        lbann_callback_summary summary_cb(&summarizer, 25);
        dnn->add_callback(&summary_cb);
        // lbann_callback_io io_cb({0});
        // dnn->add_callback(&io_cb);

        dnn->setup();

        if (grid.Rank() == 0) {
	        cout << "Layer initialized:" << endl;
                for (uint n = 0; n < dnn->get_layers().size(); n++)
                  cout << "\tLayer[" << n << "]: " << dnn->get_layers()[n]->NumNeurons << endl;
            cout << endl;
        }

        if (grid.Rank() == 0) {
	        cout << "Parameter settings:" << endl;
            cout << "\tBlock size: " << perfParams.BlockSize << endl;
            cout << "\tEpochs: " << trainParams.EpochCount << endl;
            cout << "\tMini-batch size: " << trainParams.MBSize << endl;
            // if(trainParams.MaxMBCount == 0) {
            //   cout << "\tMini-batch count (max): " << "unlimited" << endl;
            // }else {
            //   cout << "\tMini-batch count (max): " << trainParams.MaxMBCount << endl;
            // }
            cout << "\tLearning rate: " << trainParams.LearnRate << endl;
            cout << "\tEpoch count: " << trainParams.EpochCount << endl << endl;
            if(perfParams.MaxParIOSize == 0) {
              cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
            }else {
              cout << "\tMax Parallel I/O Fetch: " << perfParams.MaxParIOSize << endl;
            }
            cout << "\tDataset: " << trainParams.DatasetRootDir << endl;
        }

        // load parameters from file if available
        if (trainParams.LoadModel && trainParams.ParameterDir.length() > 0) {
          dnn->load_from_file(trainParams.ParameterDir);
        }

        ///////////////////////////////////////////////////////////////////
        // load ImageNet label list file
        ///////////////////////////////////////////////////////////////////
#if 0
        CImageNet imagenet;
        if (!imagenet.loadList(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_TrainDir,
                               trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_ValLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_ValDir,
                               trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_TestDir)) {
            cout << "ImageNet list file error: " << grid.Rank() << endl;
            return -1;
        }
        if (grid.Rank() == 0) {
            cout << "ImageNet training/validating/testing list loaded: ";
            cout << imagenet.getNumTrainData() << ", ";
            cout << imagenet.getNumValData()   << ", ";
            cout << imagenet.getNumTestData()  << endl;
            cout << endl;
        }
        /* Limit the number to training data samples to the size of
           the data set or the user-specified maximum */
        int numTrainData;
        if(trainParams.MaxMBCount != 0 && trainParams.MaxMBCount * trainParams.MBSize < imagenet.getNumTrainData()) {
          numTrainData = trainParams.MaxMBCount * trainParams.MBSize;
        }else {
          numTrainData = imagenet.getNumTrainData();
        }
        int numValData;
        if(trainParams.MaxValidationSamples != 0 && trainParams.MaxValidationSamples < imagenet.getNumValData()) {
          numValData = trainParams.MaxValidationSamples;
        }else {
          numValData = imagenet.getNumValData();
        }
        int numTestData;
        if(trainParams.MaxTestSamples != 0 && trainParams.MaxTestSamples < imagenet.getNumTestData()) {
          numTestData = trainParams.MaxTestSamples;
        }else {
          numTestData = imagenet.getNumTestData();
        }
        int MBCount = numTrainData / trainParams.MBSize;
        if (grid.Rank() == 0) {
          cout << "Processing " << numTrainData << " ImageNet training images in " << MBCount << " batches." << endl;
        }
#endif
        mpi::Barrier(grid.Comm());


        ///////////////////////////////////////////////////////////////////
        // main loop for training/testing
        ///////////////////////////////////////////////////////////////////

        int last_layer_size;
        last_layer_size = netParams.Network[netParams.Network.size()-1];

        //************************************************************************
        // read training state from checkpoint file if we have one
        //************************************************************************
        int epochStart = 0; // epoch number we should start at
        int trainStart; // index into indices we should start at

        //************************************************************************
        // mainloop for train/validate
        //************************************************************************
        for (int epoch = epochStart; epoch < trainParams.EpochCount; epoch++) {

            // TODO: need to save this in checkpoint?
            decayIterations = 1;

            //************************************************************************
            // training epoch loop
            //************************************************************************

            dnn->train(1, true);

            dnn->evaluate();
        }

        delete dnn;
    }
    catch (lbann_exception& e) { lbann_report_exception(e, comm); }
    catch (exception& e) { ReportException(e); } /// Elemental exceptions

    // free all resources by El and MPI
    Finalize();

    return 0;
}

















#if 0
int main(int argc, char* argv[])
{
    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);

    try {
        ///////////////////////////////////////////////////////////////////
        // initalize grid, block
        ///////////////////////////////////////////////////////////////////
        TrainingParams trainParams("/p/lscratchf/brainusr/datasets/ILSVRC2012/");
        PerformanceParams perfParams;
        // Read in the user specified network topology
        NetworkParams netParams;
        // Get some environment variables from the launch
        SystemParams sysParams;

        // training settings
        int decayIterations = 1;

        ProcessInput();
        PrintInputReport();

        // set algorithmic blocksize
        SetBlocksize(perfParams.BlockSize);

        // create a Grid: convert MPI communicators into a 2-D process grid
        Grid grid(mpi::COMM_WORLD);
        if (grid.Rank() == 0) {
            cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
            cout << endl;
        }

        // create timer for performance measurement
        Timer timer_io;
        Timer timer_lbann;
        Timer timer_val;
        double sec_all_io = 0;
        double sec_all_lbann = 0;
        double sec_all_val = 0;

        ///////////////////////////////////////////////////////////////////
        // load training data (ImageNet)
        ///////////////////////////////////////////////////////////////////
        DataReader_ImageNet imagenet_trainset(trainParams.MBSize, true, grid.Rank()*trainParams.MBSize, parallel_io*trainParams.MBSize);
        if (!imagenet_trainset.load(trainParams.DatasetRootDir, g_MNIST_TrainImageFile, g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile)) {
          if (comm->am_world_master()) {
            cout << "ImageNet train data error" << endl;
          }
          return -1;
        }

        ///////////////////////////////////////////////////////////////////
        // load testing data (ImageNet)
        ///////////////////////////////////////////////////////////////////
        DataReader_MNIST imagenet_testset(trainParams.MBSize, true, grid.Rank()*trainParams.MBSize, parallel_io*trainParams.MBSize);
        if (!imagenet_testset.load(g_MNIST_Dir, g_MNIST_TestImageFile, g_MNIST_TestLabelFile)) {
          if (comm->am_world_master()) {
            cout << "ImageNet Test data error" << endl;
          }
          return -1;
        }

        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////
        Optimizer_factory *optimizer;
        if (trainParams.LearnRateMethod == 1) { // Adagrad
          optimizer = new Adagrad_factory(grid, trainParams.LearnRate);
        }else if (trainParams.LearnRateMethod == 2) { // RMSprop
          optimizer = new RMSprop_factory(grid/*, trainParams.LearnRate*/);
        }else {
          optimizer = new SGD_factory(grid, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
        }

        deep_neural_network *dnn = NULL;
        {
          dnn = new deep_neural_network(optimizer, trainParams.MBSize, grid);
          int NumLayers = netParams.Network.size();
          // initalize neural network (layers)
          for (int l = 0; l < (int)NumLayers; l++) {
            string networkType;
            if(l < (int)NumLayers-1) {
              networkType = "FullyConnected";
            }else {
              // Add a softmax layer to the end
              networkType = "Softmax";
            }
            dnn->add(networkType, netParams.Network[l], trainParams.ActivationType, {new dropout(trainParams.DropOut)});
          }
        }

        if (grid.Rank() == 0) {
	        cout << "Layer initialized:" << endl;
            for (uint n = 0; n < dnn->get_layers().size(); n++)
                cout << "\tLayer[" << n << "]: " << dnn->get_layers()[n]->NumNeurons << endl;
            cout << endl;
        }

        if (grid.Rank() == 0) {
	        cout << "Parameter settings:" << endl;
            cout << "\tBlock size: " << perfParams.BlockSize << endl;
            cout << "\tEpochs: " << trainParams.EpochCount << endl;
            cout << "\tMini-batch size: " << trainParams.MBSize << endl;
            if(trainParams.MaxMBCount == 0) {
              cout << "\tMini-batch count (max): " << "unlimited" << endl;
            }else {
              cout << "\tMini-batch count (max): " << trainParams.MaxMBCount << endl;
            }
            cout << "\tLearning rate: " << trainParams.LearnRate << endl;
            cout << "\tEpoch count: " << trainParams.EpochCount << endl << endl;
            if(perfParams.MaxParIOSize == 0) {
              cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
            }else {
              cout << "\tMax Parallel I/O Fetch: " << perfParams.MaxParIOSize << endl;
            }
            cout << "\tDataset: " << trainParams.DatasetRootDir << endl;
        }

        // load parameters from file if available
        if (trainParams.LoadModel && trainParams.ParameterDir.length() > 0) {
          dnn->load_from_file(trainParams.ParameterDir);
        }

        ///////////////////////////////////////////////////////////////////
        // load ImageNet label list file
        ///////////////////////////////////////////////////////////////////
#if 0
        CImageNet imagenet;
        if (!imagenet.loadList(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_TrainDir,
                               trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_ValLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_ValDir,
                               trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_TestDir)) {
            cout << "ImageNet list file error: " << grid.Rank() << endl;
            return -1;
        }
        if (grid.Rank() == 0) {
            cout << "ImageNet training/validating/testing list loaded: ";
            cout << imagenet.getNumTrainData() << ", ";
            cout << imagenet.getNumValData()   << ", ";
            cout << imagenet.getNumTestData()  << endl;
            cout << endl;
        }
        /* Limit the number to training data samples to the size of
           the data set or the user-specified maximum */
        int numTrainData;
        if(trainParams.MaxMBCount != 0 && trainParams.MaxMBCount * trainParams.MBSize < imagenet.getNumTrainData()) {
          numTrainData = trainParams.MaxMBCount * trainParams.MBSize;
        }else {
          numTrainData = imagenet.getNumTrainData();
        }
        int numValData;
        if(trainParams.MaxValidationSamples != 0 && trainParams.MaxValidationSamples < imagenet.getNumValData()) {
          numValData = trainParams.MaxValidationSamples;
        }else {
          numValData = imagenet.getNumValData();
        }
        int numTestData;
        if(trainParams.MaxTestSamples != 0 && trainParams.MaxTestSamples < imagenet.getNumTestData()) {
          numTestData = trainParams.MaxTestSamples;
        }else {
          numTestData = imagenet.getNumTestData();
        }
        int MBCount = numTrainData / trainParams.MBSize;
        if (grid.Rank() == 0) {
          cout << "Processing " << numTrainData << " ImageNet training images in " << MBCount << " batches." << endl;
        }
#endif
        mpi::Barrier(grid.Comm());


        ///////////////////////////////////////////////////////////////////
        // main loop for training/testing
        ///////////////////////////////////////////////////////////////////

        int last_layer_size;
        last_layer_size = netParams.Network[netParams.Network.size()-1];

        // create a local matrix on each process for holding an input image
        Mat X_local(netParams.Network[0] + 1, trainParams.MBSize);
        Mat Y_local(last_layer_size, trainParams.MBSize);
        // create a distributed matrix on each process for input and output that stores the data on a single root node
        CircMat Xs(netParams.Network[0] + 1, trainParams.MBSize, grid);
        CircMat X(netParams.Network[0] + 1, 1, grid);
        CircMat XP(netParams.Network[0] + 1, 1, grid);

        CircMat Ys(last_layer_size, trainParams.MBSize, grid);
        CircMat Y(last_layer_size, 1, grid);
        CircMat YP(last_layer_size, 1, grid);

        vector<int> indices(numTrainData);

        // create a buffer for image data
        unsigned char* imagedata = new unsigned char[g_ImageNet_Width * g_ImageNet_Height * 3];

        //************************************************************************
        // read training state from checkpoint file if we have one
        //************************************************************************
        int epochStart; // epoch number we should start at
        int trainStart; // index into indices we should start at
        //bool restarted = restartShared(&epochStart, &trainStart, indices, trainParams, dnn);
        bool restarted = restartShared(&epochStart, &trainStart, indices, trainParams, dnn);
        if (! restarted) {
            // brand new run, set both starting values to 0
            epochStart = 0;
            trainStart = 0;

            // Note: libelemental intializes model params above with random values
            // seed the random number generator
            std::srand(trainParams.RandomSeed + 0);

            // Create a random ordering of the training set
            int tmpNumTrainData = numTrainData; //imagenet.getNumTrainData()-1;
            vector<int> trainingSet(tmpNumTrainData /*imagenet.getNumTrainData()-1*/);
            for (int n = 0; n < tmpNumTrainData/*imagenet.getNumTrainData()-1*/; n++) {
                trainingSet[n] = n;
            }
            if(trainParams.ShuffleTrainingData) {
                std::random_shuffle(trainingSet.begin(), trainingSet.end());
            }

            // select the first N from the randomly ordered training samples - initialize indices
            for (int n = 0; n < numTrainData; n++) {
                indices[n] = trainingSet[n];
            }
        }

        //************************************************************************
        // mainloop for train/validate
        //************************************************************************
        for (int epoch = epochStart; epoch < trainParams.EpochCount; epoch++) {
            if (grid.Rank() == 0) {
                cout << "-----------------------------------------------------------" << endl;
                cout << "[" << epoch << "] Epoch (learning rate = " << trainParams.LearnRate << ")"<< endl;
                cout << "-----------------------------------------------------------" << endl;
            }

            if (!restarted) {
              ((SoftmaxLayer*)dnn->get_layers()[dnn->get_layers().size()-1])->resetCost();
            //              dnn->Softmax->resetCost();
            }

            // TODO: need to save this in checkpoint?
            decayIterations = 1;

            //************************************************************************
            // training epoch loop
            //************************************************************************
            if (! restarted) {
                // randomly shuffle indices into training data at start of each epoch
                std::srand(trainParams.RandomSeed + epoch);
	        std::random_shuffle(indices.begin(), indices.end());
            }

            // Determine how much parallel I/O
            int TargetMaxIOSize = 1;
            if (perfParams.MaxParIOSize > 0) {
                TargetMaxIOSize = (grid.Size() < perfParams.MaxParIOSize) ? grid.Size() : perfParams.MaxParIOSize;
            }

            // if (grid.Rank() == 0) {
            //   cout << "\rTraining:      " << endl; //flush;
            // }

            int trainOffset = trainStart;
            while (trainOffset < numTrainData) {
                Zero(X_local);
                Zero(Y_local);

                // assume each reader can fetch a whole minibatch of training data
                int trainBlock = TargetMaxIOSize * trainParams.MBSize;
                int trainRemaining = numTrainData - trainOffset;
                if (trainRemaining < trainBlock) {
                    // not enough training data left for all readers to fetch a full batch
                    // compute number of readers needed
                    trainBlock = trainRemaining;
                }

                // How many parallel I/O streams can be fetched
                int IO_size = ceil((double)trainBlock / trainParams.MBSize);

                if (trainParams.EnableProfiling && grid.Rank() == 0) {
                  timer_io.Start();
                }

                // read training data/label mini batch
                if (grid.Rank() < IO_size) {
                  int myOffset = trainOffset + (grid.Rank() * trainParams.MBSize);
                  int numImages = std::min(trainParams.MBSize, numTrainData - myOffset);
                  getTrainDataMB(imagenet, &indices[myOffset], imagedata, X_local, Y_local, numImages, netParams.Network[0]);
                }
                mpi::Barrier(grid.Comm());

                if (grid.Rank() == 0) {
                  if (trainParams.EnableProfiling) {
                    sec_all_io += timer_io.Stop();
                    timer_lbann.Start();
                  }
                  cout << "\rTraining: " << trainOffset << endl;
                  //                  cout << "\b\b\b\b\b" << setw(5) << trainOffset << flush;
                  //                  cout << "\r" << setw(5) << trainOffset << "\t" << std::flush;
                  //                  cout << "\t" << setw(5) << trainOffset << "\t" << std::flush;
                  //                  flush(cout);
#if 0
                  {
                  float progress = 0.0;
                  while (progress < 1.0) {
                    int barWidth = 70;

                    std::cout << "[";
                    int pos = barWidth * progress;
                    for (int i = 0; i < barWidth; ++i) {
                      if (i < pos) std::cout << "=";
                      else if (i == pos) std::cout << ">";
                      else std::cout << " ";
                    }
                    std::cout << "] " << int(progress * 100.0) << " %\r";
                    std::cout.flush();

                    progress += 0.16; // for demonstration only
                  }
                  std::cout << std::endl;
                  }
#endif
                }

                // train mini batch
                for(int r = 0; r < IO_size; r++) {
                  Zero(Xs);
                  Zero(Ys);
                  Xs.SetRoot(r);
                  Ys.SetRoot(r);
                  //if (grid.Rank() == r) {
                  //  Xs.CopyFromRoot(X_local);
                  //  Ys.CopyFromRoot(Y_local);
                  //}else {
                  //  Xs.CopyFromNonRoot();
                  //  Ys.CopyFromNonRoot();
                  //}
                  //mpi::Barrier(grid.Comm());
                  if (grid.Rank() == r) {
                      CopyFromRoot(X_local, Xs);
                      CopyFromRoot(Y_local, Ys);
                  }else {
                      CopyFromNonRoot(Xs);
                      CopyFromNonRoot(Ys);
                  }


                  dnn->train(Xs, Ys, trainParams.LearnRate, trainParams.LearnRateMethod);

#if 0
                  if(/*n*/trainOffset + r * trainParams.MBSize > decayIterations * trainParams.LrDecayCycles) {
                    trainParams.LearnRate = trainParams.LearnRate * trainParams.LrDecayRate;
                    decayIterations++;
                    if(grid.Rank() == 0) {
                      cout << "Changing the learning rate to " << trainParams.LearnRate << " after processing " << (/*n*/trainOffset + r * trainParams.MBSize) << " dataums" << endl;
                    }
                  }
#endif
                  mpi::Barrier(grid.Comm());
                }
                if (trainParams.EnableProfiling && grid.Rank() == 0) {
                  sec_all_lbann += timer_lbann.Stop();
                }
                // Finished training on single pass of data
                mpi::Barrier(grid.Comm());

                // increment our offset into the training data
                trainOffset += trainBlock;

                //************************************************************************
                // checkpoint our training state
                //************************************************************************
                // TODO: checkpoint
                bool ckpt_epoch = ((trainOffset == numTrainData) && (trainParams.Checkpoint > 0) && (epoch % trainParams.Checkpoint == 0));
                if (trainParams.SaveModel && trainParams.ParameterDir.length() > 0 && ckpt_epoch)
                {
                    checkpoint(epoch, trainOffset, indices, trainParams, dnn);
                    checkpointShared(epoch, trainOffset, indices, trainParams, dnn);
                }
            }

            // reset our training offset for the next epoch
            restarted = false;
            trainStart = 0;

            if (grid.Rank() == 0) {
                cout << " ... done" << endl;
                if (trainParams.EnableProfiling) {
                  double sec_all_total = sec_all_io + sec_all_lbann;

                  double sec_mbatch_io = sec_all_io / (MBCount * (epoch+1));
                  double sec_mbatch_lbann = sec_all_lbann / (MBCount * (epoch+1));
                  double sec_mbatch_total = (sec_all_io + sec_all_lbann) / (MBCount * (epoch+1));

                  double sec_each_io = sec_mbatch_io / trainParams.MBSize;
                  double sec_each_lbann = sec_mbatch_lbann / trainParams.MBSize;
                  double sec_each_total = (sec_mbatch_io + sec_mbatch_lbann) / trainParams.MBSize;

                  double avg_cost = ((SoftmaxLayer*)dnn->get_layers()[dnn->get_layers().size()-1])->avgCost();
                    //                    double avg_cost = dnn->Softmax->avgCost();
                  cout << "Average Softmax Cost: " << avg_cost << endl;
                  cout << "#, Host, Nodes, Processes, Cores, TasksPerNode, Epoch, Training Samples, Mini-Batch Size, Mini-Batch Count, Total Time, Total I/O, Total lbann, MB Time, MB I/O, MB lbann, Sample Time, Sample I/O, Sample lbann" << endl;
                  cout << "# [RESULT], " << sysParams.HostName << ", " << sysParams.NumNodes << ", " << grid.Size() << ", " << sysParams.NumCores << ", " << sysParams.TasksPerNode << ", " << epoch << ", ";
                  cout << numTrainData << ", " << trainParams.MBSize << ", " << MBCount << ", ";
                  cout << sec_all_total    << ", " << sec_all_io    << ", " << sec_all_lbann    << ", ";
                  cout << sec_mbatch_total << ", " << sec_mbatch_io << ", " << sec_mbatch_lbann << ", ";
                  cout << sec_each_total   << ", " << sec_each_io   << ", " << sec_each_lbann   << endl;
#if 0
                  cout << "Training time (sec): ";
                  cout << "total: "      << sec_all_total    << " (I/O: " << sec_all_io    << ", lbann: " << sec_all_lbann    << ")" << endl;
                  cout << "mini-batch: " << sec_mbatch_total << " (I/O: " << sec_mbatch_io << ", lbann: " << sec_mbatch_lbann << ")" << endl;
                  cout << "each: "       << sec_each_total   << " (I/O: " << sec_each_io   << ", lbann: " << sec_each_lbann   << ")" << endl;
                  cout << endl;
#endif
                }
            }

#if 1
            // Update the learning rate on each epoch
            trainParams.LearnRate = trainParams.LearnRate * trainParams.LrDecayRate;
            if(grid.Rank() == 0) {
              cout << "Changing the learning rate to " << trainParams.LearnRate << " after processing " << (epoch+1) << " epochs" << endl;
            }
#endif

            //************************************************************************
            // validating/testing loop
            //************************************************************************
            int numTopOneErrors = 0, numTopFiveErrors = 0;
            double sumerrors = 0;
            if (trainParams.EnableProfiling && grid.Rank() == 0) {
              timer_val.Start();
            }
            for (int n = 0; n < numValData; n++) {

                // read validating data/label
                int imagelabel;
                if (grid.Rank() == 0) {
                  if(trainParams.TestWithTrainData) {
                    getTrainData(imagenet, indices[n], imagedata, X, Y, netParams.Network[0]);
                    for(int i = 0; i < Y.Height(); i++) {
                      if(Y.GetLocal(i,0) == 1) {
                        imagelabel = i;
                      }
                    }
                  }else {
                    getValData(imagenet, n, imagedata, X, imagelabel, netParams.Network[0]);
                  }
                }
                mpi::Barrier(grid.Comm());

                {
                    // test dnn
                    dnn->test(X, Y);

                    // validate
                    if (grid.Rank() == 0) {
                        int labelidx[5] = {-1, -1, -1, -1, -1};
                        double labelmax[5] = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
                        //                        cout << endl;
                        for (int m = 0; m < netParams.Network[netParams.Network.size()-1]; m++) {
                          for(int k = 0; k < 5; k++) {
                            if (labelmax[k] <= Y.GetLocal(m, 0)) {
                                for(int i = 4; i > k; i--) {
                                  labelmax[i] = labelmax[i-1];
                                  labelidx[i] = labelidx[i-1];
                                }
                                labelmax[k] = Y.GetLocal(m, 0);
                                labelidx[k] = m;
                                break;
                            }
                          }
                        }
                        if (imagelabel != labelidx[0]) {
                            numTopOneErrors++;
                        }
                        bool topFiveMatch = false;
                        for(int i = 0; i < 5; i++) {
                          if(imagelabel == labelidx[i]) {
                            topFiveMatch = true;
                            break;
                          }
                        }
                        if(!topFiveMatch) {
                          numTopFiveErrors++;
                        }
                            // Print(Y);
#if 0
                        if(!topFiveMatch) {
                          cout << "\rTesting: " << n << "th sample, " << numTopOneErrors << " top one errors and " << numTopFiveErrors
                               << " top five errors - image label " << imagelabel << " =?= ";
                          for(int i = 0; i < 5; i++) {
                            cout << labelidx[i] << "(" << labelmax[i] << ") ";
                          }
                          cout << endl;
                          int bad_val = 0;
                          for(int i = 0; i < Y.Height(); i++) {
                            if(Y.GetLocal(i,0) < 0.00001) {
                              bad_val++;
                            }else {
                              cout << i << "=" << Y.GetLocal(i,0) << " ";
                            }
                          }
                          cout << endl;
                          cout << bad_val << " insignificant values"<< endl << endl;
                        }
#endif
                    }
                }
            }
            if (grid.Rank() == 0) {
                if (trainParams.EnableProfiling) {
                  sec_all_val += timer_val.Stop();
                }
                cout << endl;
                if (trainParams.EnableProfiling) {
                  //                  double sec_all_vall_total = sec_all_io + sec_all_lbann;

                  double sec_val_each_total = sec_all_val / (numValData * (epoch+1));

                  cout << "Validation time (sec): ";
                  cout << "total: "      << sec_all_val << endl;
                  cout << "each: "       << sec_val_each_total << endl;
                  cout << endl;
                }
            }

            float topOneAccuracy = (float)(numValData - numTopOneErrors) / numValData * 100.0f;
            float topFiveAccuracy = (float)(numValData - numTopFiveErrors) / numValData * 100.0f;
            if (grid.Rank() == 0) {
              cout << "Top One Accuracy:  " << topOneAccuracy << "%" << endl;
              cout << "Top Five Accuracy: " << topFiveAccuracy << "%" << endl << endl;
            }

            //************************************************************************
            // checkpoint our training state
            //************************************************************************
/*
            if (trainParams.SaveModel && trainParams.ParameterDir.length() > 0 &&
                trainParams.Checkpoint > 0 && (epoch % trainParams.Checkpoint == 0))
            {
                checkpoint(epoch+1, trainParams, dnn);
            }
*/
        }
        delete [] imagedata;

        // save final model parameters
        if (trainParams.SaveModel && trainParams.ParameterDir.length() > 0) {
          dnn->save_to_file(trainParams.ParameterDir);
        }

        delete dnn;
    }
    catch (exception& e) { ReportException(e); }

    // free all resources by El and MPI
    Finalize();

    return 0;
}
#endif
