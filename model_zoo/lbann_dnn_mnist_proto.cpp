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

#include "lbann/callbacks/lbann_callback_dump_weights.hpp"
#include "lbann/callbacks/lbann_callback_dump_activations.hpp"
#include "lbann/callbacks/lbann_callback_dump_gradients.hpp"
#include "lbann/lbann.hpp"
#include "lbann/proto/lbann_proto_common.hpp"

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

    try {

        ///////////////////////////////////////////////////////////////////
        // initalize grid, block
        ///////////////////////////////////////////////////////////////////

        // Initialize parameter defaults
        TrainingParams trainParams;
        //trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/MNIST/";
        trainParams.EpochCount = 20;
        //trainParams.MBSize = 128;
        trainParams.LearnRate = 0.01;
        trainParams.DropOut = -1.0f;
        trainParams.ProcsPerModel = 0;
        //trainParams.PercentageTrainingSamples = 0.90;
        //trainParams.PercentageValidationSamples = 1.00;
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

        //get input prototext filenames
        string prototext_model_fn = Input("--prototext_fn", "prototext model filename", "none");
        string prototext_dr_fn = Input("--prototext_dr_fn", "prototext data reader filename", "none");
        //if (prototext_model_fn == "none" or prototext_dr_fn == "none") {
        if (prototext_dr_fn == "none" or prototext_model_fn == "none") {
          if (comm->am_world_master()) {
            cerr << endl << __FILE__ << " " << __LINE__ << " :: error - you must use "
                 << " --prototext_fn and --prototext_dr_fn to supply prototext filenames\n\n";
          }
          Finalize();
          return 9;
        }
        int mini_batch_size = Input("--mb-size", "mini_batch_size", 0);

        lbann_data::LbannPB pb;
        lbann_data::LbannPB pb_reader;
        readPrototextFile(prototext_model_fn.c_str(), pb);
        readPrototextFile(prototext_dr_fn.c_str(), pb_reader);


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
        std::map<execution_mode, DataReader*> data_readers;
        init_data_readers(comm->am_world_master(), pb_reader, data_readers, mini_batch_size);
        if (comm->am_world_master()) {
          for (auto it : data_readers) {
            cerr << "data reader; role: " << it.second->get_role() << " num data: " << it.second->getNumData() << endl;
          }
        }

        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////

        // Initialize optimizer
        optimizer_factory *optimizer_fac = init_optimizer_factory(comm, pb);
        #if 0
        if (trainParams.LearnRateMethod == 1) { // Adagrad
          optimizer_fac = new adagrad_factory(comm, trainParams.LearnRate);
        } else if (trainParams.LearnRateMethod == 2) { // RMSprop
          optimizer_fac = new rmsprop_factory(comm, trainParams.LearnRate);
        } else if (trainParams.LearnRateMethod == 3) { // Adam
          optimizer_fac = new adam_factory(comm, trainParams.LearnRate);
        } else {
          optimizer_fac = new sgd_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
        }
        #endif

        // Initialize network
        layer_factory* lfac = new layer_factory();
        deep_neural_network dnn(mini_batch_size, comm, new objective_functions::categorical_cross_entropy(comm), lfac, optimizer_fac);
        dnn.add_metric(new metrics::categorical_accuracy(comm));

        //first layer
        input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, mini_batch_size, data_readers);
        dnn.add(input_layer);
        
        //second layer
        dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, 100, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});

        //third layer
        dnn.add("FullyConnected", data_layout::MODEL_PARALLEL, 30, trainParams.ActivationType, weight_initialization::glorot_uniform, {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});

        //fourth layer
        dnn.add("Softmax", data_layout::MODEL_PARALLEL, 10, activation_type::ID, weight_initialization::glorot_uniform, {});

        //fifth layer
        target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, mini_batch_size, data_readers, true);
        dnn.add(target_layer);

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
          cout << "\tMini-batch size: " << mini_batch_size << endl;
          cout << "\tLearning rate: " << trainParams.LearnRate << endl;
          cout << "\tEpoch count: " << trainParams.EpochCount << endl << endl;
        }

        ///////////////////////////////////////////////////////////////////
        // main loop for training/testing
        ///////////////////////////////////////////////////////////////////

        // Initialize the model's data structures
        dnn.setup();

        // set checkpoint directory and checkpoint interval
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
        delete optimizer_fac;
        delete comm;
    }
    catch (lbann_exception& e) { lbann_report_exception(e, comm); }
    catch (exception& e) { ReportException(e); } /// Elemental exceptions

    // free all resources by El and MPI
    Finalize();

    return 0;
}
