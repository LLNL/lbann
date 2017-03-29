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
#include "lbann/callbacks/lbann_callback_dump_weights.hpp"
#include "lbann/callbacks/lbann_callback_dump_activations.hpp"
#include "lbann/callbacks/lbann_callback_dump_gradients.hpp"
#include "lbann/lbann.hpp"
#include "lbann/proto/lbann_proto.hpp"

// for read/write
#include <unistd.h>

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
    lbann_comm* comm = NULL;

    lbann_proto *pb = lbann_proto::get();

    try {

      const string prototext_fn = Input("--prototext_fn", "filename for writing a prototext file; default is 'none,' in which case no file will be written", std::string("none"));

        // Get data files
        const string g_MNIST_TrainLabelFile = Input("--train-label-file",
                                                    "MNIST training set label file",
                                                    std::string("train-labels-idx1-ubyte"));
        const string g_MNIST_TrainImageFile = Input("--train-image-file",
                                                    "MNIST training set image file",
                                                    std::string("train-images-idx3-ubyte"));
        const string g_MNIST_TestLabelFile = Input("--test-label-file",
                                                   "MNIST test set label file",
                                                   std::string("t10k-labels-idx1-ubyte"));
        const string g_MNIST_TestImageFile = Input("--test-image-file",
                                                   "MNIST test set image file",
                                                   std::string("t10k-images-idx3-ubyte"));

        //determine if we're going to scale, subtract mean, etc;
        //scaling/standardization is on a per-example basis (computed independantly
        //for each image)
        bool scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
        bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", false);
        bool unit_variance = Input("--unit-variance", "standardize to unit-variance", false);

        //if set to true, above three settings have no effect
        bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

        ///////////////////////////////////////////////////////////////////
        // initalize grid, block
        ///////////////////////////////////////////////////////////////////

        // Initialize parameter defaults
        TrainingParams trainParams;
        trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/MNIST/";
        trainParams.EpochCount = 20;
        trainParams.MBSize = 128;
        trainParams.LearnRate = 0.01;
        trainParams.DropOut = -1.0f;
        trainParams.ProcsPerModel = 0;
        trainParams.PercentageTrainingSamples = 0.90;
        trainParams.PercentageValidationSamples = 1.00;
        PerformanceParams perfParams;
        perfParams.BlockSize = 256;

        //NetworkParams network_params;
        //SystemParams system_params;

        // Parse command-line inputs
        trainParams.parse_params();
        perfParams.parse_params();
        //network_params.parse_params();
        //system_params.parse_params();

        ProcessInput();
        PrintInputReport();

        //register params with the lbann_proto class
        //pb->add_network_params(network_params);
        pb->add_performance_params(perfParams);
        //pb->add_system_params(system_params);
        pb->add_training_params(trainParams);


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

        // Initialize lbann with the communicator.
        lbann::initialize(comm);
        init_random(42);
        init_data_seq_random(42);

        //tell the lbann_proto class who is the master
        if (comm->am_world_master()) {
          pb->set_master(true);
        } else {
          pb->set_master(false);
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
        lbann_proto::data_reader_params d1;
        d1.name = "mnist";
        d1.role = "train";
        d1.mini_batch_size = trainParams.MBSize;
        d1.shuffle = true;
        d1.root_dir = trainParams.DatasetRootDir;
        d1.data_filename = g_MNIST_TrainImageFile;
        d1.label_filename = g_MNIST_TrainLabelFile;
        d1.percent_samples = trainParams.PercentageTrainingSamples;
        pb->add_data_reader(d1);

        DataReader_MNIST mnist_trainset(trainParams.MBSize, true);
        mnist_trainset.set_file_dir(trainParams.DatasetRootDir);
        mnist_trainset.set_data_filename(g_MNIST_TrainImageFile);
        mnist_trainset.set_label_filename(g_MNIST_TrainLabelFile);
        mnist_trainset.set_use_percent(trainParams.PercentageTrainingSamples);
        mnist_trainset.load();

        mnist_trainset.scale(scale);
        mnist_trainset.subtract_mean(subtract_mean);
        mnist_trainset.unit_variance(unit_variance);
        mnist_trainset.z_score(z_score);

        if (comm->am_world_master()) {
          cout << "Training using " << (trainParams.PercentageTrainingSamples*100) << "% of the training data set, which is " << mnist_trainset.getNumData() << " samples." << endl;
        }

        ///////////////////////////////////////////////////////////////////
        // create a validation set from the unused training data (MNIST)
        ///////////////////////////////////////////////////////////////////
        DataReader_MNIST mnist_validation_set(mnist_trainset); // Clone the training set object
        if (!mnist_validation_set.swap_used_and_unused_index_sets()) { // Swap the used and unused index sets so that it validates on the remaining data
          if (comm->am_world_master()) {
            cerr << __FILE__ << " " << __LINE__ << " MNIST validation data error" << endl;
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
        lbann_proto::data_reader_params d2;
        d2.name = "mnist";
        d2.role = "test";
        d2.mini_batch_size = trainParams.MBSize;
        d2.shuffle = true;
        d2.root_dir = trainParams.DatasetRootDir;
        d2.data_filename = g_MNIST_TestImageFile;
        d2.label_filename = g_MNIST_TestLabelFile;
        d2.percent_samples = trainParams.PercentageTestingSamples;
        pb->add_data_reader(d2);

        DataReader_MNIST mnist_testset(trainParams.MBSize, true);
        mnist_testset.set_file_dir(trainParams.DatasetRootDir);
        mnist_testset.set_data_filename(g_MNIST_TestImageFile);
        mnist_testset.set_label_filename(g_MNIST_TestLabelFile);
        mnist_testset.set_use_percent(trainParams.PercentageTestingSamples);
        mnist_testset.load();


        //@TODO: add to lbann_proto.hpp
        mnist_testset.scale(scale);
        mnist_testset.subtract_mean(subtract_mean);
        mnist_testset.unit_variance(unit_variance);
        mnist_testset.z_score(z_score);

        if (comm->am_world_master()) {
          cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << mnist_testset.getNumData() << " samples." << endl;
        }

        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////

        // Initialize optimizer
        lbann_proto::optimizer_params o1;
        o1.learn_rate = trainParams.LearnRate;
        o1.momentum = 0.9;
        o1.decay = trainParams.LrDecayRate;
        o1.nesterov = false;

        Optimizer_factory *optimizer;
        if (trainParams.LearnRateMethod == 1) { // Adagrad
          optimizer = new Adagrad_factory(comm, trainParams.LearnRate);
          o1.name = "adagrad";
        }else if (trainParams.LearnRateMethod == 2) { // RMSprop
          optimizer = new RMSprop_factory(comm/*, trainParams.LearnRate*/);
          o1.name = "rms";
        } else if (trainParams.LearnRateMethod == 3) { // Adam
          optimizer = new Adam_factory(comm, trainParams.LearnRate);
          o1.name = "adam";
        }else {
          optimizer = new SGD_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
          o1.name = "sgd";
        }
        pb->add_optimizer(o1);

        // Initialize network
        lbann_proto::model_params m1;
        m1.name = "dnn";
        m1.objective_function = "categorical_cross_entropy";
        m1.mini_batch_size = trainParams.MBSize;
        m1.num_epochs = trainParams.EpochCount;
        m1.add_metric("categorical_accuracy");
        pb->add_model(m1);

        layer_factory* lfac = new layer_factory();
        deep_neural_network dnn(trainParams.MBSize, comm, new objective_functions::categorical_cross_entropy(comm), lfac, optimizer);
        dnn.add_metric(new metrics::categorical_accuracy(comm));
        std::map<execution_mode, DataReader*> data_readers = {std::make_pair(execution_mode::training,&mnist_trainset), 
                                                               std::make_pair(execution_mode::validation, &mnist_validation_set), 
                                                               std::make_pair(execution_mode::testing, &mnist_testset)};

        //input_layer *input_layer = new input_layer_distributed_minibatch(comm,  (int) trainParams.MBSize, data_readers);
        
        //first layer
        lbann_proto::layer_params layer_1;
        layer_1.name = "input_distributed_minibatch_parallel_io";
        layer_1.mini_batch_size = trainParams.MBSize;
        layer_1.num_parallel_readers = parallel_io;
        pb->add_layer(layer_1);

        input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, (int) trainParams.MBSize, data_readers);
        dnn.add(input_layer);
        
        //second layer
        lbann_proto::layer_params layer_2;
        layer_2.name = "fully_connected";
        layer_2.num_prev_neurons = dnn.num_previous_neurons();
        layer_2.num_neurons = 100;
        layer_2.activation = trainParams.ActivationType;
        layer_2.weight_init = weight_initialization::glorot_uniform;
        lbann_proto::regularizer_params r1;
        r1.name = "dropout";
        r1.dropout = trainParams.DropOut;
        layer_2.add_regularizer(r1);
        pb->add_layer(layer_2);

        dnn.add("FullyConnected", 100, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(comm, trainParams.DropOut)});

        //third layer
        lbann_proto::layer_params layer_3;
        layer_3.name = "fully_connected";
        layer_3.num_prev_neurons = dnn.num_previous_neurons();
        layer_3.num_neurons = 30;
        layer_3.activation = trainParams.ActivationType;
        layer_3.weight_init = weight_initialization::glorot_uniform;
        pb->add_layer(layer_3);
        dnn.add("FullyConnected", 30, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(comm, trainParams.DropOut)});

        //fourth layer
        lbann_proto::layer_params layer_4;
        layer_4.name = "softmax";
        layer_4.num_prev_neurons = dnn.num_previous_neurons();
        layer_4.num_neurons = 10;
        layer_4.activation = activation_type::ID;
        layer_4.weight_init = weight_initialization::glorot_uniform;
        pb->add_layer(layer_4);
        dnn.add("Softmax", 10, activation_type::ID, weight_initialization::glorot_uniform, {});

        //fifth layer
        lbann_proto::layer_params layer_5;
        layer_5.name = "target_distributed_minibatch_parallel_io";
        layer_5.mini_batch_size = trainParams.MBSize;
        layer_5.num_parallel_readers = parallel_io;
        layer_5.shared_data_reader = true;
        layer_5.for_regression = false;
        pb->add_layer(layer_5);
        target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, (int) trainParams.MBSize, data_readers, true);
        dnn.add(target_layer);

        //callbacks @TODO: add to lbann_proto
        lbann_callback_print print_cb;
        dnn.add_callback(&print_cb);
        lbann_callback_dump_weights* dump_weights_cb;
        lbann_callback_dump_activations* dump_activations_cb;
        lbann_callback_dump_gradients* dump_gradients_cb;
        if (trainParams.DumpWeights) {
          dump_weights_cb = new lbann_callback_dump_weights(
            trainParams.DumpDir);
          dnn.add_callback(dump_weights_cb);
        }
        if (trainParams.DumpActivations) {
          dump_activations_cb = new lbann_callback_dump_activations(
            trainParams.DumpDir);
          dnn.add_callback(dump_activations_cb);
        }
        if (trainParams.DumpGradients) {
          dump_gradients_cb = new lbann_callback_dump_gradients(
            trainParams.DumpDir);
          dnn.add_callback(dump_gradients_cb);
        }
        // lbann_callback_io io_cb({0,3});
        // dnn.add_callback(&io_cb);
        //lbann_callback_io io_cb({0,3});
        //        dnn.add_callback(&io_cb);
        //lbann_callback_debug debug_cb(execution_mode::testing);
        //        dnn.add_callback(&debug_cb);

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

        // Initialize the model's data structures
        dnn.setup();

        if (comm->am_world_master()) {
          bool success = pb->writePrototextFile(prototext_fn.c_str());
          if (success) {
            cout << "prototext file written to: " << prototext_fn << endl;
          } else {
            cout << "prototext file NOT written; you must pass --prototext_fn <string>\n"
                 << "to write a file\n";
          }
        }

        // set checkpoint directory and checkpoint interval
        // @TODO: add to lbann_proto
        dnn.set_checkpoint_dir(trainParams.ParameterDir);
        dnn.set_checkpoint_epochs(trainParams.CkptEpochs);
        dnn.set_checkpoint_steps(trainParams.CkptSteps);
        dnn.set_checkpoint_secs(trainParams.CkptSecs);

        // restart model from checkpoint if we have one
        dnn.restartShared();

        // train/test
        while (dnn.get_cur_epoch() < trainParams.EpochCount) {
#if 0
            // optionally check gradients
            if (n > 0 && n % 10000 == 0) {
               printf("Checking gradients...\n");
               double errors[g_NumLayers];
               dnn.checkGradient(Xs, Ys, errors);
               printf("gradient errors: ");
               for (uint l = 0; l < g_NumLayers; l++)
                   printf("%lf ", errors[l]);
               printf("\n");
            }
#endif

            dnn.train(1, true);

            // Update the learning rate on each epoch
            // trainParams.LearnRate = trainParams.LearnRate * trainParams.LrDecayRate;
            // if(grid.Rank() == 0) {
            //   cout << "Changing the learning rate to " << trainParams.LearnRate << " after processing " << (t+1) << " epochs" << endl;
            // }

            // testing
            int numerrors = 0;

            dnn.evaluate(execution_mode::testing);
        }

        // Free dynamically allocated memory
        // delete target_layer;  // Causes segfault
        // delete input_layer;  // Causes segfault
        // delete lfac;  // Causes segfault
        if (trainParams.DumpWeights) {
          delete dump_weights_cb;
        }
        if (trainParams.DumpActivations) {
          delete dump_activations_cb;
        }
        if (trainParams.DumpGradients) {
          delete dump_gradients_cb;
        }
        delete optimizer;
        delete comm;
    }
    catch (lbann_exception& e) { lbann_report_exception(e, comm); }
    catch (exception& e) { ReportException(e); } /// Elemental exceptions

    // free all resources by El and MPI
    Finalize();

    return 0;
}
