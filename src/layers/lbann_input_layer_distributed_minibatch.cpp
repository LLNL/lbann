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

#include "lbann/layers/lbann_input_layer_distributed_minibatch.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;


lbann::input_layer_distributed_minibatch::input_layer_distributed_minibatch(lbann_comm* comm, uint mini_batch_size, DataReader *training_data_reader, DataReader *testing_data_reader, std::vector<regularizer*> regs)
  : input_layer(comm, mini_batch_size, training_data_reader, testing_data_reader, regs), Xs(comm->get_model_grid())
{
  //  Index = index;
  m_root = 0;
}

void lbann::input_layer_distributed_minibatch::setup(int num_prev_neurons) {
  Layer::setup(num_prev_neurons);
  if(m_training_data_reader != NULL) {
    m_training_data_reader->setup(0, m_mini_batch_size);
  }

  if(m_testing_data_reader != NULL) {
    m_testing_data_reader->setup(0, m_mini_batch_size);
  }

  Zeros(*Acts, NumNeurons + 1, m_mini_batch_size);
  Zeros(X_local, NumNeurons + 1, m_mini_batch_size);
}

void lbann::input_layer_distributed_minibatch::fp_linearity(
  ElMat&, ElMat&, ElMat&, ElMat&) {
  DataReader *data_reader = input_layer::select_data_reader();

  if (comm->get_rank_in_model() == m_root) {
    Zero(X_local);
    data_reader->fetch_data(X_local);
    /// Set the bias term in the last row of the input matrix
    int linear_data_size = data_reader->get_linearized_data_size();
    for(size_t n = 0; n < m_mini_batch_size; n++) {
      X_local.Set(linear_data_size, n, 1);
    }
  }

  if (comm->get_rank_in_model() == m_root) {
    CopyFromRoot(X_local, Xs);
  }else {
    CopyFromNonRoot(Xs);
  }

  comm->model_barrier();

  Copy(Xs, *Acts);
}

/**
 * Once a mini-batch is processed, resuffle the data for the next batch if necessary
 */
bool lbann::input_layer_distributed_minibatch::update() {
  DataReader *data_reader = input_layer::select_data_reader();
  return !data_reader->update();
}
