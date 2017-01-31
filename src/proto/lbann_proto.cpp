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

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

using google::protobuf::io::FileOutputStream;

using namespace std;
using namespace lbann;


lbann_proto * lbann_proto::s_instance = new lbann_proto;

lbann_proto::lbann_proto() {}

lbann_proto::~lbann_proto() {
  delete s_instance;
}

void lbann_proto::writePrototextFile(const char *fn) {
  int fd = open(fn, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  if (not google::protobuf::TextFormat::Print(m_pb, output)) {
    stringstream err;
    err << __FILE__ << " " << __LINE__ << " lbann_proto::writePrototextFile() - failed to write prototext file: " << fn;
    throw runtime_error(err.str());
  }
  delete output;
  close(fd);
}


void lbann_proto::DataReaderMNIST_ctor(int batch_size, bool shuffle) {
    allocateDataReader();
    lbann_data::DataReader *reader = m_pb.mutable_data_reader();
    lbann_data::DataReaderMnist *mnist = reader->add_mnist();
    int size = reader->mnist_size();
    if (size == 1) {
      mnist->set_role("train");
    } else if (size == 2) {
      mnist->set_role("test");
    } else {
      std::cerr << __FILE__<<" "<<__LINE__<<" lbann_proto ERROR\n";
      exit(-1);
    }
    mnist->set_batch_size(batch_size);
    mnist->set_shuffle(shuffle);
  }

void lbann_proto::DataReaderMNIST_load(std::string file_dir, std::string image_file, std::string label_file) {
    lbann_data::DataReader *reader = m_pb.mutable_data_reader();
    int size = reader->mnist_size();
    lbann_data::DataReaderMnist *mnist = reader->mutable_mnist(size-1); //@TODO add error check
    mnist->set_file_dir(file_dir);
    mnist->set_image_file(image_file);
    mnist->set_label_file(label_file);
}

void lbann_proto::allocateDataReader() {
    if (not m_pb.has_data_reader()) {
      lbann_data::DataReader *reader = new lbann_data::DataReader;
      m_pb.set_allocated_data_reader(reader);
    }
}

void lbann_proto::Model_ctor(std::string name, std::string objective_function, std::string optimizer) {
    lbann_data::Model *model;
    if (not m_pb.has_model()) {
      model = new lbann_data::Model;
      m_pb.set_allocated_model(model);
    }
    model->set_name(name);
    model->set_objective_function(objective_function);
    model->set_optimizer(optimizer);
}

void lbann_proto::Model_train(int num_epochs, int evaluation_frequency) {
    lbann_data::Model *model = m_pb.mutable_model();
    model->set_num_epochs(num_epochs);
    model->set_evaluation_frequency(evaluation_frequency);
}

void lbann_proto::Layer_InputDistributedMiniBatchParallelIO_ctor(int num_parallel_readers, int mini_batch_size) {
  lbann_data::Model *model = m_pb.mutable_model();
  lbann_data::Layer *layer = model->add_layer();
  lbann_data::InputDistributedMiniBatchParallelIO *real_layer = new lbann_data::InputDistributedMiniBatchParallelIO;
  layer->set_allocated_input_distributed_minibatch_parallel_io(real_layer);
  real_layer->set_num_parallel_readers(num_parallel_readers);
  real_layer->set_mini_batch_size(mini_batch_size);
}

void lbann_proto::Layer_FullyConnected_ctor(
    int num_prev_neurons,
    int num_neurons,
    int mini_batch_size,
    std::string activation_type,
    std::string weight_initialization,
    std::string optimizer) {
  lbann_data::Model *model = m_pb.mutable_model();
  lbann_data::Layer *layer = model->add_layer();
  lbann_data::FullyConnected *real_layer = new lbann_data::FullyConnected;
  layer->set_allocated_fully_connected(real_layer);
  real_layer->set_num_prev_neurons(num_prev_neurons);
  real_layer->set_num_neurons(num_neurons);
  real_layer->set_mini_batch_size(mini_batch_size);
  real_layer->set_activation_type(activation_type);
  real_layer->set_weight_initialization(weight_initialization);
  real_layer->set_optimizer(optimizer);
}

