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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/lbann_input_layer_distributed_minibatch_parallel_io.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::input_layer_distributed_minibatch_parallel_io::input_layer_distributed_minibatch_parallel_io(lbann_comm *comm, int num_parallel_readers, uint mini_batch_size, std::map<execution_mode, DataReader*> data_readers, std::vector<regularizer*> regs)
  : input_layer(comm, mini_batch_size, data_readers, regs),
    distributed_minibatch_parallel_io(comm, num_parallel_readers, mini_batch_size, data_readers),
    Xs(comm->get_model_grid())
{

  m_type = layer_type::input_distributed_minibatch;
}

void lbann::input_layer_distributed_minibatch_parallel_io::setup(int num_prev_neurons) {
  input_layer::setup(num_prev_neurons);
  if(io_layer::m_data_sets_span_models) {
    int stride = Layer::comm->get_num_models() * m_num_parallel_readers_training * Layer::m_mini_batch_size;
    int base_offset = Layer::comm->get_rank_in_model() * Layer::comm->get_num_models() * Layer::m_mini_batch_size;
    int model_offset = Layer::comm->get_model_rank() * Layer::m_mini_batch_size;
    //cout << "["<< Layer::comm->get_rank_in_world() << "] Setting up input layer, with " << Layer::comm->get_num_models() << " models and " << m_num_parallel_readers_training << " parallel readers and " << Layer::m_mini_batch_size << " mb size, which gives a stride of " << stride << " and my model offset is " << model_offset << " and my base offset is " << base_offset /*(Layer::comm->get_rank_in_model() * Layer::m_mini_batch_size)*/ << endl;
    io_layer::setup_data_readers_for_training(base_offset,
                                              stride,
                                              model_offset);
    /// Note that the data readers for evaluation should not be partitioned over multiple models (otherwise each model will be scored on a different set of data)
    io_layer::setup_data_readers_for_evaluation(Layer::comm->get_rank_in_model() * Layer::m_mini_batch_size,
                                                m_num_parallel_readers_training * Layer::m_mini_batch_size);
  }else {
    io_layer::setup_data_readers_for_training(Layer::comm->get_rank_in_model() * Layer::m_mini_batch_size,
                                              m_num_parallel_readers_training * Layer::m_mini_batch_size);
    io_layer::setup_data_readers_for_evaluation(Layer::comm->get_rank_in_model() * Layer::m_mini_batch_size,
                                                m_num_parallel_readers_training * Layer::m_mini_batch_size);
  }

  Zeros(*m_activations, NumNeurons, Layer::m_mini_batch_size);
  Zeros(X_local, NumNeurons, Layer::m_mini_batch_size);

  m_local_data_valid = false;
  m_local_reader_done = false;
  m_num_data_per_epoch = 0;
}

void lbann::input_layer_distributed_minibatch_parallel_io::fp_linearity() {
  DataReader *data_reader = input_layer::select_data_reader();
  int num_parallel_readers = get_num_parallel_readers();

  int num_samples_in_batch = fetch_to_local_matrix(X_local);
  if(is_current_root()) {
    /// Only update the number of samples processed by this parallel reader, when it is the current root
    input_layer::update_num_samples_processed(num_samples_in_batch);
  }

  /// Let each rank know this size of the current mini-batch 
  /// Note that this field has to be updated before distributing the data
  neural_network_model->set_current_mini_batch_size(Layer::comm->model_broadcast(m_root, num_samples_in_batch));

  distribute_from_local_matrix(X_local, Xs);

  Copy(Xs, *m_activations);
}

/**
 * Once a mini-batch is processed, resuffle the data for the next batch if necessary
 */
bool lbann::input_layer_distributed_minibatch_parallel_io::update() {
  return is_data_set_processed();
}


int lbann::input_layer_distributed_minibatch_parallel_io::fetch_from_data_reader(Mat &M_local) {
  DataReader *data_reader = input_layer::select_data_reader();
  return data_reader->fetch_data(M_local);
}

void lbann::input_layer_distributed_minibatch_parallel_io::preprocess_data_samples(Mat& M_local, int num_samples_in_batch) {
  return;
}

bool lbann::input_layer_distributed_minibatch_parallel_io::update_data_reader() {
  DataReader *data_reader = input_layer::select_data_reader();
  return data_reader->update();
}

execution_mode lbann::input_layer_distributed_minibatch_parallel_io::get_execution_mode() {
  return m_execution_mode;
}

Mat* lbann::input_layer_distributed_minibatch_parallel_io::get_local_mat() {
  return &X_local;
}

CircMat* lbann::input_layer_distributed_minibatch_parallel_io::get_dist_mat() {
  return &Xs;
}
