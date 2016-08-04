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
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::input_layer_distributed_minibatch_parallel_io::input_layer_distributed_minibatch_parallel_io(lbann_comm *comm, int num_parallel_readers, uint mini_batch_size, DataReader *training_data_reader, DataReader *testing_data_reader, std::vector<regularizer*> regs)
  : input_layer(comm, mini_batch_size, training_data_reader, testing_data_reader, regs), 
    distributed_minibatch_parallel_io(comm, num_parallel_readers, mini_batch_size, training_data_reader->getNumData(), testing_data_reader->getNumData()),
    Xs(comm->get_model_grid())
{
}

void lbann::input_layer_distributed_minibatch_parallel_io::setup(int num_prev_neurons) {
  io_layer::setup_data_readers(Layer::comm->get_rank_in_model() * Layer::m_mini_batch_size,
                               m_num_parallel_readers_training * Layer::m_mini_batch_size);

  Zeros(*Acts, NumNeurons + 1, Layer::m_mini_batch_size);
  Zeros(X_local, NumNeurons + 1, Layer::m_mini_batch_size);

  m_local_data_valid = false;
  m_local_reader_done = false;
  m_num_data_per_epoch = 0;
}

void lbann::input_layer_distributed_minibatch_parallel_io::fp_linearity(
  ElMat&, ElMat&, ElMat&, ElMat&) {
  DataReader *data_reader = input_layer::select_data_reader();
  int num_parallel_readers = get_num_parallel_readers();

  int num_samples_in_batch = fetch_to_local_matrix(X_local);
  input_layer::update_num_samples_processed(num_samples_in_batch);

  distribute_from_local_matrix(X_local, Xs);

  Copy(Xs, *Acts);
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
  DataReader *data_reader = input_layer::select_data_reader();
  /// Set the bias term in the last row of the input matrix
  int linear_data_size = data_reader->get_linearized_data_size();
  for(int n = 0; n < num_samples_in_batch; n++) {
    M_local.Set(linear_data_size, n, 1);
  }
}

bool lbann::input_layer_distributed_minibatch_parallel_io::update_data_reader() {
  DataReader *data_reader = input_layer::select_data_reader();
  return data_reader->update();
}
  
execution_mode lbann::input_layer_distributed_minibatch_parallel_io::get_execution_mode() {
  return m_execution_mode;
}
