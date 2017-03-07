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

#include "lbann/data_readers/lbann_data_reader_mnist.hpp"
#include "lbann/callbacks/lbann_callback_dump_weights.hpp"
#include "lbann/callbacks/lbann_callback_dump_activations.hpp"
#include "lbann/callbacks/lbann_callback_dump_gradients.hpp"
#include "lbann/lbann.hpp"
#include "lbann/proto/lbann_proto.hpp"

int main(int argc, char* argv[])
{}
#if 0

// for read/write
//#include <unistd.h>

using namespace std;
using namespace lbann;
using namespace El;

/*
 * unlike all other classes in Lbann -- except for lbann_proto --
 * this driver directly manipulates protobuffer objects
 */

//============================================================================
// the following are defined after the end of main();
// possibly they should be refactored into separate files,
//============================================================================
void init_training_params(
    const lbann_data::LbannPB &p,
    TrainingParams &train_params);

void init_performance_params(
    const lbann_data::LbannPB &p,
    PerformanceParams &performance_params);

void init_network_params(
    const lbann_data::LbannPB &p,
    NetworkParams &network_params);

void init_system_params(
    const lbann_data::LbannPB &p,
    SystemParams &system_params);

void init_data_readers(
    bool master,
    const lbann_data::LbannPB &p,
    std::map<execution_mode, DataReader*> &data_readers,
    const TrainingParams &train_params);

void init_model(
    bool master,
    const lbann_data::LbannPB &p,
    sequential_model *&model,
    const TrainingParams &train_params,
    lbann_comm* comm);

void add_layers(
    bool master,
    const lbann_data::LbannPB &p,
    sequential_model *model,
    lbann_comm* comm,
    int parallel_io,
    const TrainingParams &train_params,
    std::map<execution_mode, DataReader*> &data_readers);

activation_type get_activation_type(string s);

weight_initialization get_weight_initialization_type(string s);

//============================================================================
int main(int argc, char* argv[])
{
    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);
    init_random(42);
    lbann_comm* comm = NULL;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool master = rank == 0 ? true : false;

    try {
      stringstream err;

        //get input prototext filename
        string prototext_fn = Input("--prototext_fn", "prototext filename", "none");
        if (prototext_fn == "none" and master) {
            err << __FILE__ << " " << __LINE__ << " :: error - you must use --prototext_fn"
                << " to supply a prototext filename";
            throw lbann_exception(err.str());
        }

        ProcessInput();
        PrintInputReport();

        //read prototext file, and get outermost prototext container 
        lbann_proto *pb = lbann_proto::get();
        pb->readPrototextFile(prototext_fn.c_str());
        lbann_data::LbannPB &p = pb->getLbannPB();

        //initialize Lbann Params classes
        TrainingParams train_params;
        PerformanceParams performance_params;
        NetworkParams network_params;
        SystemParams system_params;
        init_training_params(p, train_params);
        init_performance_params(p, performance_params);
        init_network_params(p, network_params);
        init_system_params(p, system_params);

        //initialize data readers
        std::map<execution_mode, DataReader*> data_readers;
        init_data_readers(master, p, data_readers, train_params);
        // Set algorithmic blocksize
        SetBlocksize(performance_params.BlockSize);

        // Set up the communicator and get the grid.
        comm = new lbann_comm(train_params.ProcsPerModel);
        Grid& grid = comm->get_model_grid();
        if (comm->am_world_master()) {
            cout << "Number of models: " << comm->get_num_models() << endl
                 << "Grid is " << grid.Height() << " x " << grid.Width() << endl << endl;
        }

        // set and report parallel io param
        int parallel_io = performance_params.MaxParIOSize;
        if (parallel_io == 0) {
            parallel_io = comm->get_procs_per_model();
            if (comm->am_world_master()) {
                cout << "\tMax Parallel I/O Fetch: " << parallel_io
                     << " (Limited to # Processes)" << endl;
            }
        } else {
            if (comm->am_world_master()) {
                cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
            }
        }

        //the following set parallel_io = 1 at this point; per convo with
        //brian ( 7feb2017 ) this is probably left over junk from a previous
        //incarnation -- so should be safe to ignore
        //model_zoo/lbann_alexnet.cpp
        //model_zoo/lbann_dnn_imagenet.cpp
        //model_zoo/lbann_dnn_multi_imagenet.cpp
        //model_zoo/lbann_greedy_layerwise_autoencoder_imagenet.cpp

        cudnn::cudnn_manager* cudnn = NULL;
//@TODO not sure how to handle this ... probably add s
#if __LIB_CUDNN
//        cudnn::cudnn_manager* cudnn = new cudnn::cudnn_manager(comm);
#endif // __LIB_CUDNN

        // construct the model and add the layers
        sequential_model *model = 0;
        init_model(master, p, model, train_params, comm);
        add_layers(master, p, model, comm, parallel_io, train_params, data_readers);

        //@TODO: need to add metrics and callbacks to lbann.proto
        metrics::categorical_accuracy *x = new metrics::categorical_accuracy(comm);
        x->neural_network_model = model;
        model->add_metric(x);
        //model->add_metric(new metrics::categorical_accuracy(comm));
        lbann_callback_print print_cb;
        model->add_callback(&print_cb);

        if (comm->am_world_master()) {
          cout << endl << "Calling model->setup()\n";
        }
        model->setup();

        // set checkpoint directory and checkpoint interval @TODO

        // restart model from checkpoint if we have one @TODO

        if (comm->am_world_master()) {
            cout << "Parameter settings:" << endl;
            cout << "\tMini-batch size: " << train_params.MBSize << endl;
            cout << "\tLearning rate: " << train_params.LearnRate << endl;
            cout << "\tEpoch count: " << train_params.EpochCount << endl;
        }

        // train/test
        while (model->get_cur_epoch() < train_params.EpochCount) {
            model->train(1, true);
            model->evaluate(execution_mode::testing);
        }

    } catch (lbann_exception& e) {
        lbann_report_exception(e, comm);
    } catch (exception& e) {
        ReportException(e);  /// Elemental exceptions
    }

    // free all resources by El and MPI
    Finalize();

    return 0;
}

//=============================================================================

void init_training_params(const lbann_data::LbannPB &p, TrainingParams &train_params)
{
    const lbann_data::TrainingParams &t = p.training_params();
    train_params.EnableProfiling = t.enable_profiling();
    train_params.RandomSeed = PB_FIX(t.random_seed());
    train_params.ShuffleTrainingData = PB_FIX(t.shuffle_training_data());
    train_params.PercentageTrainingSamples = PB_FIXD(t.percentage_training_samples());
    train_params.PercentageValidationSamples = PB_FIXD(t.percentage_validation_samples());
    train_params.PercentageTestingSamples = PB_FIXD(t.percentage_testing_samples());
    train_params.TestWithTrainData = PB_FIX(t.test_with_train_data());
    train_params.EpochStart = PB_FIX(t.epoch_start());
    train_params.EpochCount = PB_FIX(t.epoch_count());
    train_params.MBSize = PB_FIX(t.mb_size());
    train_params.LearnRate = PB_FIXD(t.learn_rate());
    train_params.LearnRateMethod = PB_FIX(t.learn_rate_method());
    train_params.LrDecayRate = PB_FIXD(t.lr_decay_rate());
    train_params.LrDecayCycles = PB_FIX(t.lr_decay_cycles());
    train_params.LrMomentum = PB_FIXD(t.lr_momentum());
    train_params.DropOut = PB_FIXD(t.dropout());
    train_params.Lambda = PB_FIXD(t.lambda());
    train_params.DatasetRootDir = t.dataset_root_dir();
    train_params.SaveImageDir = t.save_image_dir();
    train_params.ParameterDir = t.parameter_dir();
    train_params.SaveModel = t.save_model();
    train_params.LoadModel = t.load_model();
    train_params.CkptEpochs = PB_FIX(t.ckpt_epochs());
    train_params.CkptSteps = PB_FIX(t.ckpt_steps());
    train_params.CkptSecs = PB_FIX(t.ckpt_secs());
    train_params.TrainFile = t.train_file();
    train_params.TestFile = t.test_file();
    train_params.SummaryDir = t.summary_dir();
    train_params.DumpWeights = t.dump_weights();
    train_params.DumpActivations = t.dump_activations();
    train_params.DumpGradients = t.dump_gradients();
    train_params.DumpDir = t.dump_dir();
    train_params.IntermodelCommMethod = PB_FIX(t.intermodel_comm_method());
    train_params.ProcsPerModel = PB_FIX(t.procs_per_model());

    string a = t.activation_type();
    train_params.ActivationType = get_activation_type(a);

    string b = t.weight_initialization();
    train_params.WeightInitType = get_weight_initialization_type(b);
}

void init_performance_params(const lbann_data::LbannPB &p, PerformanceParams &performance_params)
{
    const lbann_data::PerformanceParams &t = p.performance_params();
  performance_params.BlockSize = PB_FIX(t.block_size());
  performance_params.MaxParIOSize = PB_FIX(t.max_par_io_size());
}

void init_network_params(const lbann_data::LbannPB &p, NetworkParams &network_params)
{
    const lbann_data::NetworkParams &t = p.network_params();
  network_params.NetworkStr = t.network_str();
  //network_params.parse_network_string(); //@TODO is private
  //
}

void init_system_params(const lbann_data::LbannPB &p, SystemParams &system_params)
{
    const lbann_data::SystemParams &t = p.system_params();
  system_params.HostName = t.host_name();
  system_params.NumNodes = PB_FIX(t.num_nodes());
  system_params.NumCores = PB_FIX(t.num_cores());
  system_params.TasksPerNode = PB_FIX(t.tasks_per_node());
}

void init_data_readers(bool master, const lbann_data::LbannPB &p, std::map<execution_mode, DataReader*> &data_readers, const TrainingParams &train_params)
{
    stringstream err;
    const lbann_data::DataReader &data_reader = p.data_reader();

    //determine the reader type
    enum {NONE, MNIST, CIFAR10, IMAGENET, NCI, NCI_REGRESSION};
    unordered_map<int, int> reader_types;

    if (data_reader.mnist_size()) {
        reader_types[MNIST] = data_reader.mnist_size();
    }
    if (data_reader.cifar10_size()) {
        reader_types[CIFAR10] = data_reader.cifar10_size();
    }
    if (data_reader.imagenet_size()) {
        reader_types[IMAGENET] = data_reader.imagenet_size();
    }
    if (data_reader.nci_size()) {
        reader_types[NCI] = data_reader.nci_size();
    }
    if (data_reader.nci_regression_size()) {
        reader_types[NCI_REGRESSION] = data_reader.nci_regression_size();
    }

    //sanity checks
    if (reader_types.size() > 1 and master) {
        err << __FILE__ << " " << __LINE__ << " :: error: multiple DataReader types; should e.g, only be DataReaderMnist or DataReaderCifar10; something appears to be wrong with your prototext file";
        throw lbann_exception(err.str());
    }
    if (reader_types.size() == 0 and master) {
        err << __FILE__ << " " << __LINE__ << " :: error: no DataReader type specified in your prototext file";
        throw lbann_exception(err.str());
    }

    unordered_map<int,int>::iterator t = reader_types.begin();
    int tp = t->first;
    int num_readers = t->second;

    //@TODO: add in code for remaining types
    switch(tp) {
    case MNIST :
        for (int j=0; j<num_readers; j++) {
            const ::lbann_data::DataReaderMnist &mnist = data_reader.mnist(j);
            string role = mnist.role();

            double percent = (role == "train") ? train_params.PercentageTrainingSamples : train_params.PercentageTestingSamples;
            DataReader_MNIST *data_set = new DataReader_MNIST(mnist.batch_size(), mnist.shuffle());
            if (not data_set->load(mnist.file_dir(), mnist.image_file(), mnist.label_file(), percent)) {
                err <<  __FILE__ << " " << __LINE__ << " :: error: DataReader_MNIST.load() failed";
                throw lbann_exception(err.str());
            }

            //add to data_readers map; also error check that role is either "train" or "test"
            if (role == "train") {
                data_readers[execution_mode::training] = data_set;
            } else if (role == "test") {
                data_readers[execution_mode::testing] = data_set;
            } else if (master) {
                err << __FILE__ << " " << __LINE__ << " :: error: DataReaderMnist role= " << mnist.role()
                    << " should be either 'train' or 'test'";
                throw lbann_exception(err.str());
            }

            if (role == "train") {
                DataReader_MNIST *validation_set = new DataReader_MNIST(*data_set);
                if (not validation_set->swap_used_and_unused_index_sets() and master) {
                    err << __FILE__ << " " << __LINE__ << " :: MNIST validation data error";
                    throw lbann_exception(err.str());
                }
                if (train_params.PercentageValidationSamples == 1.00) {
                    if (master) {
                        cout << "Validating training using " << ((1.00 - train_params.PercentageTrainingSamples)*100) << "% of the training data set, which is " << validation_set->getNumData() << " samples." << endl;
                    }
                } else {
                    size_t preliminary_validation_set_size = validation_set->getNumData();
                    size_t final_validation_set_size = validation_set->trim_data_set(train_params.PercentageValidationSamples);
                    if (master) {
                        cout << "Trim the validation data set from " << preliminary_validation_set_size << " samples to " << final_validation_set_size << " samples." << endl;
                    }
                }
                data_readers[execution_mode::validation] = validation_set;
            }
        }
        break;

    default :
        err << __FILE__ << " " << __LINE__ << " error: data reader type; error may be caused by lack of implementation (TODO)";
        throw lbann_exception(err.str());
    } //switch tp
}


void init_model(bool master, const lbann_data::LbannPB &p, sequential_model *&model, const TrainingParams &train_params, lbann_comm* comm)
{
    stringstream err;

    // Initialize optimizer factory
    Optimizer_factory *optimizer;
    if (train_params.LearnRateMethod == 1) { // Adagrad
        optimizer = new Adagrad_factory(comm, train_params.LearnRate);
        if (master) cout << "optimizer is: Adagrad; learnRate: " << train_params.LearnRate << "\n";
    } else if (train_params.LearnRateMethod == 2) { // RMSprop
        optimizer = new RMSprop_factory(comm/*, train_params.LearnRate*/);
        if (master) cout << "optimizer is: RMSprop\n";
    } else {
        optimizer = new SGD_factory(comm, train_params.LearnRate, train_params.LrMomentum, train_params.LrDecayRate, true); 
        if (master) cout << "optimizer is: SGD; learnRate: " << train_params.LearnRate
                         << " momentum: " << train_params.LrMomentum
                         << " decay: " << train_params.LrDecayRate << endl;
      //@TODO last param is 'bool nesterov' I've no idea what that is/means;
      //should probably be added to TrainingParams class
    }


    const lbann_data::Model &m = p.model();
    const string &name = m.name();

    layer_factory* lfac = new layer_factory();

    // construct the appropriate objective function
    objective_functions::objective_fn* obj_fn;
    const string &obj_name = m.objective_function();
    if (obj_name == "categorical_cross_entropy") {
        obj_fn =  new objective_functions::categorical_cross_entropy(comm);
    } else if (obj_name == "mean_squared_error") {
        obj_fn =  new objective_functions::mean_squared_error(comm);
    } else {
      if (master) {
        err << "unknown name for objective function: " << obj_name;
        throw lbann_exception(err.str());
      }  
    }

    //@TODO add code for  other two types: greedy_layerwise_autoencoder,
    //      stacked_autoencoder
    if (name == "dnn") {
        model = new deep_neural_network(
            train_params.MBSize,
            comm,
            obj_fn,
            lfac,
            optimizer);
    } else {
        err << "unknown model name: " << name << " possibly the code to handle this hasn't been implemented, but should be";
        throw lbann_exception(err.str());
    }
}

//==========================================================================
// start of layer-specific add functions
//==========================================================================

activation_type get_activation_type(string s)
{
    if (s == "sigmoid") return activation_type::SIGMOID;
    if (s == "tanh") return activation_type::TANH;
    if (s == "relu") return activation_type::RELU;
    if (s == "id") return activation_type::ID;
    if (s == "leaky_relu") return activation_type::LEAKY_RELU;
    if (s == "smooth_relu") return activation_type::SMOOTH_RELU;
    if (s == "elu") return activation_type::ELU;
}

weight_initialization get_weight_initialization_type(string s)
{
  if (s == "zero") return weight_initialization::zero;
  if (s == "uniform") return weight_initialization::uniform;
  if (s == "normal") return weight_initialization::normal;
  if (s == "glorot_normal") return weight_initialization::glorot_normal;
  if (s == "glorot_uniform") return weight_initialization::glorot_uniform;
  if (s == "he_normal") return weight_initialization::he_normal;
  if (s == "he_uniform") return weight_initialization::he_uniform;
}

void add_input_distributed_minibatch(
    const lbann_data::InputDistributedMiniBatch &layer,
    sequential_model *model,
    bool master,
    lbann_comm* comm,
    const TrainingParams &train_params,
    std::map<execution_mode, DataReader*> &data_readers)
{
    stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << " not yet implemented";
    throw lbann_exception(err.str());
}

void add_pooling(
    const lbann_data::Pooling &layer,
    sequential_model *model,
    bool master,
    lbann_comm* comm,
    const TrainingParams &train_params,
    std::map<execution_mode, DataReader*> &data_readers)
{
    stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << " not yet implemented";
    throw lbann_exception(err.str());
}

void add_convolution(
    const lbann_data::Convolution &layer,
    sequential_model *model,
    bool master,
    lbann_comm* comm,
    const TrainingParams &train_params,
    std::map<execution_mode, DataReader*> &data_readers)
{
    stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << " not yet implemented";
    throw lbann_exception(err.str());
}

void add_target_parallel(
    const lbann_data::TargetParallel &layer,
    sequential_model *model,
    bool master,
    lbann_comm* comm,
    const TrainingParams &train_params,
    std::map<execution_mode, DataReader*> &data_readers)
{
    stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << " not yet implemented";
    throw lbann_exception(err.str());
}

void add_target_distributed_minibatch(
    const lbann_data::TargetDistributedMinibatch &layer,
    sequential_model *model,
    bool master,
    lbann_comm* comm,
    const TrainingParams &train_params,
    std::map<execution_mode, DataReader*> &data_readers)
{
    stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << " not yet implemented";
    throw lbann_exception(err.str());
}

//==========================================================================
// end of layer-specific add functions
//==========================================================================
//
void add_layers(
  bool master, 
  const lbann_data::LbannPB &p, 
  sequential_model *model, 
  lbann_comm* comm, 
  int parallel_io, 
  const TrainingParams &train_params, 
  std::map<execution_mode, DataReader*> &data_readers)
{
    stringstream err;

    const lbann_data::Model &m = p.model();
    int num_layers = m.layer_size();
    for (int i=0; i<num_layers; i++) {
        const ::lbann_data::Layer& layer = m.layer(i);

        //determine the layer type
        int layer_count = 0;
        if (layer.has_input_distributed_minibatch_parallel_io()) {
            ++layer_count;
            input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(comm, parallel_io, train_params.MBSize, data_readers);
            model->add(input_layer);
            if (master) cout << "add_layers(): adding input_layer_distributed_minibatch_parallel_io\n";
        }

        if (layer.has_input_distributed_minibatch()) {
            ++layer_count;
            add_input_distributed_minibatch(layer.input_distributed_minibatch(), model, master, comm, train_params, data_readers);
            if (master) cout << "add_layers(): adding input_layer_distributed_minibatch\n";
        }

        if (layer.has_fully_connected()) {
            ++layer_count;
            const lbann_data::FullyConnected &f = layer.fully_connected();
            model->add(
                "FullyConnected",
                f.num_neurons(),
                get_activation_type(f.activation_type()),
                get_weight_initialization_type(f.weight_initialization()),
            {new dropout(comm, train_params.DropOut)}); //@TODO - fix regularizers!
            if (master) cout << "add_layers(): adding FullyConnected\n";
        }

        if (layer.has_pooling()) {
            ++layer_count;
            add_pooling(layer.pooling(), model, master, comm, train_params, data_readers);
        }

        if (layer.has_convolution()) {
            ++layer_count;
            add_convolution(layer.convolution(), model, master, comm, train_params, data_readers);
        }

        if (layer.has_softmax()) {
            ++layer_count;
            const lbann_data::Softmax &f = layer.softmax();
            model->add(
                "Softmax",
                f.num_neurons(),
                activation_type::ID,
                get_weight_initialization_type(f.weight_initialization()),
                {});
            if (master) cout << "add_layers(): adding Softmax\n";
        }

        if (layer.has_target_parallel()) {
            ++layer_count;
            add_target_parallel(layer.target_parallel(), model, master, comm, train_params, data_readers);
        }

        if (layer.has_target_distributed_minibatch()) {
            ++layer_count;
            add_target_distributed_minibatch(layer.target_distributed_minibatch(), model, master, comm, train_params, data_readers);
        }

        if (layer.has_target_distributed_minibatch_parallel_io()) {
            ++layer_count;
            const lbann_data::TargetDistributedMinibatchParallelIO &f = layer.target_distributed_minibatch_parallel_io();
            target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(comm, parallel_io, (int)train_params.MBSize, data_readers, true);
            if (master) cout << "add_layers(): adding target_layer_distributed_minibatch_parallel_io\n";
            //@TODO last argument is "for_regression" - what to do???
            model->add(target_layer);
        }

        if (layer_count != 1) {
            err << __FILE__ << " " << __LINE__ << " :: error: layer_count = "
                << layer_count << "; should be one; indicates error in prototext file.";
            throw lbann_exception(err.str());
        }
    }
}
#endif
