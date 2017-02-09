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

#include "lbann/proto/lbann_proto.hpp"
#include "lbann/utils/lbann_exception.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <mpi.h>

using google::protobuf::io::FileOutputStream;

using namespace std;
using namespace lbann;

string get_activation_type(activation_type s);
string get_weight_initialization_type(weight_initialization s);

lbann_proto * lbann_proto::s_instance = new lbann_proto;

lbann_proto::lbann_proto() { }

lbann_proto::~lbann_proto()
{
  delete s_instance;
}


void lbann_proto::readPrototextFile(const char *fn)
{
  stringstream err;
  int fd = open(fn, O_RDONLY);
  if (fd == -1) {
    err <<  __FILE__ << " " << __LINE__ << " :: failed to open " << fn << " for reading";
    throw lbann_exception(err.str());
  }
  google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, &m_pb);
  if (not success and m_master) {
    err <<  __FILE__ << " " << __LINE__ << " :: failed to read or parse prototext file: " << fn << endl;
    throw lbann_exception(err.str());
  }
}

bool lbann_proto::writePrototextFile(const char *fn)
{
  int fd = open(fn, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd == -1) {
    return false;
  }
  FileOutputStream* output = new FileOutputStream(fd);
  if (not google::protobuf::TextFormat::Print(m_pb, output)) {
    close(fd);
    delete output;
    return false;
  }
  delete output;
  close(fd);
  return true;
}

void lbann_proto::add_network_params(const NetworkParams &p)
{
  lbann_data::NetworkParams *t = m_pb.mutable_network_params();
  t->set_network_str(p.NetworkStr);
}

void lbann_proto::add_performance_params(const PerformanceParams &p)
{
  lbann_data::PerformanceParams *t = m_pb.mutable_performance_params();
  t->set_block_size(PB_FIX(p.BlockSize));
  t->set_max_par_io_size(PB_FIX(p.MaxParIOSize));

}

void lbann_proto::add_system_params(const SystemParams &p)
{
  lbann_data::SystemParams *t = m_pb.mutable_system_params();
  t->set_host_name(p.HostName);
  t->set_num_nodes(PB_FIX(p.NumNodes));
  t->set_num_cores(PB_FIX(p.NumCores));
  t->set_tasks_per_node(PB_FIX(p.TasksPerNode));
}

void lbann_proto::add_training_params(const TrainingParams &p)
{
  lbann_data::TrainingParams *t = m_pb.mutable_training_params();
    t->set_enable_profiling(p.EnableProfiling);
    t->set_random_seed(PB_FIX(p.RandomSeed));
    t->set_shuffle_training_data(PB_FIX(p.ShuffleTrainingData));
    t->set_percentage_training_samples(PB_FIXD(p.PercentageTrainingSamples));
    t->set_percentage_validation_samples(PB_FIXD(p.PercentageValidationSamples));
    t->set_percentage_testing_samples(PB_FIXD(p.PercentageTestingSamples));
    t->set_test_with_train_data(PB_FIX(p.TestWithTrainData));
    t->set_epoch_start(PB_FIX(p.EpochStart));
    t->set_epoch_count(PB_FIX(p.EpochCount));
    t->set_mb_size(PB_FIX(p.MBSize));
    t->set_learn_rate(PB_FIXD(p.LearnRate));
    t->set_learn_rate_method(PB_FIX(p.LearnRateMethod));
    t->set_lr_decay_rate(PB_FIXD(p.LrDecayRate));
    t->set_lr_decay_cycles(PB_FIX(p.LrDecayCycles));
    t->set_lr_momentum(PB_FIXD(p.LrMomentum));
    t->set_dropout(PB_FIXD(p.DropOut));
    t->set_lambda(PB_FIXD(p.Lambda));
    t->set_dataset_root_dir(p.DatasetRootDir);
    t->set_save_image_dir(p.SaveImageDir);
    t->set_parameter_dir(p.ParameterDir);
    t->set_save_model(p.SaveModel);
    t->set_load_model(p.LoadModel);
    t->set_ckpt_epochs(PB_FIX(p.CkptEpochs));
    t->set_ckpt_steps(PB_FIX(p.CkptSteps));
    t->set_ckpt_secs(PB_FIX(p.CkptSecs));
    t->set_train_file(p.TrainFile);
    t->set_test_file(p.TestFile);
    t->set_summary_dir(p.SummaryDir);
    t->set_dump_weights(p.DumpWeights);
    t->set_dump_activations(p.DumpActivations);
    t->set_dump_gradients(p.DumpGradients);
    t->set_dump_dir(p.DumpDir);
    t->set_intermodel_comm_method(PB_FIX(p.IntermodelCommMethod));
    t->set_procs_per_model(PB_FIX(p.ProcsPerModel));
    t->set_activation_type(get_activation_type(p.ActivationType));
    t->set_weight_initialization(get_weight_initialization_type(p.WeightInitType));
}

void lbann_proto::add_data_reader(const data_reader_params &p)
{
  stringstream err;
  lbann_data::DataReader *reader = m_pb.mutable_data_reader();

  //@TODO: add code for other data readers
  if (p.name == "mnist") {
    lbann_data::DataReaderMnist *mnist = reader->add_mnist();
    int size = reader->mnist_size();
    mnist->set_role(p.role);
    mnist->set_batch_size(PB_FIX(p.mini_batch_size));
    mnist->set_shuffle(p.shuffle);
    mnist->set_file_dir(p.root_dir);
    mnist->set_image_file(p.data_filename);
    mnist->set_label_file(p.label_filename);
    mnist->set_percent_samples(PB_FIXD(p.percent_samples));
  } else {
    err << __FILE__ << " " << __LINE__ << " :: lbann_proto::add_data_reader() - data reader with this name is not implemented: " << p.name;
    throw lbann_exception(err.str());
  }
}

void lbann_proto::add_optimizer(const optimizer_params &p)
{
  lbann_data::Model *model = m_pb.mutable_model();
  lbann_data::Optimizer *opt = model->mutable_optimizer();
  opt->set_name(p.name);
  opt->set_learn_rate(PB_FIXD(p.learn_rate));
  opt->set_momentum(PB_FIXD(p.momentum));
  opt->set_decay(PB_FIXD(p.decay));
  opt->set_nesterov(p.nesterov);
}

void lbann_proto::add_model(const model_params &p)
{
  lbann_data::Model *model = m_pb.mutable_model();
  model->set_name(p.name);
  model->set_objective_function(p.objective_function);
  model->set_mini_batch_size(PB_FIX(p.mini_batch_size));
  model->set_num_epochs(PB_FIX(p.num_epochs));
  for (size_t j = 0; j < p.metrics.size(); j++) {
    model->add_metric(p.metrics[j]);
  }
}

void lbann_proto::add_layer(const layer_params &p)
{
  lbann_data::Model *model = m_pb.mutable_model();
  lbann_data::Layer *layer = model->add_layer();
  string name = p.name;

  if (name == "input_distributed_minibatch_parallel_io") {
    lbann_data::InputDistributedMiniBatchParallelIO *real_layer = layer->mutable_input_distributed_minibatch_parallel_io();
    real_layer->set_num_parallel_readers(PB_FIX(p.num_parallel_readers));
    real_layer->set_mini_batch_size(PB_FIX(p.mini_batch_size));
  }

  else if (name == "fully_connected") {
    lbann_data::FullyConnected *real_layer = layer->mutable_fully_connected();
    real_layer->set_num_prev_neurons(PB_FIX(p.num_prev_neurons));
    real_layer->set_num_neurons(PB_FIX(p.num_neurons));
    real_layer->set_mini_batch_size(PB_FIX(p.mini_batch_size));
    real_layer->set_activation_type( get_activation_type(p.activation) );
    real_layer->set_weight_initialization( get_weight_initialization_type(p.weight_init) );
    for (size_t j=0; j<p.regularizers.size(); j++) {
      //XX real_layer->add_regularizer(p.regularizers[j]);
    }

  } else if (name == "softmax") {
    lbann_data::Softmax *real_layer = layer->mutable_softmax();
    real_layer->set_num_prev_neurons(PB_FIX(p.num_prev_neurons));
    real_layer->set_num_neurons(PB_FIX(p.num_neurons));
    real_layer->set_activation_type( get_activation_type(p.activation) );
    real_layer->set_weight_initialization( get_weight_initialization_type(p.weight_init) );
  }

  else if (name == "target_distributed_minibatch_parallel_io") {
    lbann_data::TargetDistributedMinibatchParallelIO *real_layer = layer->mutable_target_distributed_minibatch_parallel_io();
    real_layer->set_num_parallel_readers(p.num_parallel_readers);
    real_layer->set_mini_batch_size(p.mini_batch_size);
    real_layer->set_shared_data_reader(p.shared_data_reader);
    real_layer->set_for_regression(p.for_regression);
  }

  else {
    stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: lbann_proto::add_layer: not implemented or nothing known about layer type: " << name;
    throw lbann_exception(err.str());
  }
}


string get_activation_type(activation_type a)
{
  switch (a) {
  case activation_type::SIGMOID :
    return "sigmoid";
  case activation_type::TANH :
    return "tanh";
  case activation_type::RELU :
    return "relu";
  case activation_type::ID :
    return "id";
  case activation_type::LEAKY_RELU :
    return "leaky_relu";
  case activation_type::SMOOTH_RELU :
    return "smooth_relu";
  case activation_type::ELU :
    return "elu";
  default :
    return "none";
  }
}

string get_weight_initialization_type(weight_initialization a)
{
  switch (a) {
  case weight_initialization::zero :
    return "zero";
  case weight_initialization::uniform :
    return "uniform";
  case weight_initialization::normal :
    return "normal";
  case weight_initialization::glorot_normal :
    return "glorot_normal";
  case weight_initialization::glorot_uniform :
    return "glorot_uniform";
  case weight_initialization::he_normal :
    return "he_normal";
  case weight_initialization::he_uniform :
    return "he_uniform";
  default :
    return "none";
  }
}


