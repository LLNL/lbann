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

#include "lbann/layers/lbann_target_layer_distributed_minibatch.hpp"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::target_layer_distributed_minibatch::target_layer_distributed_minibatch(lbann_comm* comm, uint mini_batch_size, std::map<execution_mode, DataReader*> data_readers, bool shared_data_reader, bool for_regression)
  : target_layer(comm, mini_batch_size, data_readers, shared_data_reader, for_regression), Ys(comm->get_model_grid())
{
  m_type = layer_type::target_distributed_minibatch;
  //  Index = index;
  m_root = 0;
  //  NumNeurons = m_training_data_reader->get_linearized_label_size(); /// @todo NumNeurons should be hidden inside of an accessor function
}

void lbann::target_layer_distributed_minibatch::setup(int num_prev_neurons) {
  target_layer::setup(num_prev_neurons);
  if(!m_shared_data_reader) { /// If the target layer shares a data reader with an input layer, do not setup the data reader a second time
    if(io_layer::m_data_sets_span_models) {
      io_layer::setup_data_readers_for_training(0, Layer::comm->get_num_models() * Layer::m_mini_batch_size,
                                                Layer::comm->get_model_rank() * Layer::m_mini_batch_size);
      io_layer::setup_data_readers_for_evaluation(0, m_mini_batch_size);
    }else {
      io_layer::setup_data_readers_for_training(0, m_mini_batch_size);
      io_layer::setup_data_readers_for_evaluation(0, m_mini_batch_size);
    }
  }

  /// @todo put in warning about bad target size
  if(num_prev_neurons != NumNeurons) {
    throw -1;
  }

  Zeros(*m_error_signal, NumNeurons, m_mini_batch_size);
  Zeros(Y_local, NumNeurons, m_mini_batch_size);
  Zeros(Ys, NumNeurons, m_mini_batch_size);

}

void lbann::target_layer_distributed_minibatch::fp_linearity() {
  DataReader *data_reader = target_layer::select_data_reader();

  if (comm->get_rank_in_model() == m_root) {
    Zero(Y_local);
    data_reader->fetch_label(Y_local);
  }

  if (comm->get_rank_in_model() == m_root) {
    CopyFromRoot(Y_local, Ys);
  }else {
    CopyFromNonRoot(Ys);
  }

  comm->model_barrier();
  Copy(Ys, *m_activations);

  /// Compute and record the objective function score
  DataType avg_error = neural_network_model->obj_fn->compute_obj_fn(*m_prev_activations_v, *m_activations_v);
  neural_network_model->obj_fn->record_obj_fn(m_execution_mode, avg_error);

  int64_t curr_mini_batch_size = neural_network_model->get_current_mini_batch_size();
  for (auto&& m : neural_network_model->metrics) {
    double cur_num_errors = (int) m->compute_metric(*m_prev_activations_v, *m_activations_v);
    m->record_error(cur_num_errors, curr_mini_batch_size);
  }

  return;
}

void lbann::target_layer_distributed_minibatch::bp_linearity() {

  // Compute initial error signal
  neural_network_model->obj_fn->compute_obj_fn_derivative(m_prev_layer_type,
                                                          *m_prev_activations_v,
                                                          *m_activations_v,
                                                          *m_error_signal_v);

}

/**
 * Once a mini-batch is processed, resuffle the data for the next batch if necessary
 */
bool lbann::target_layer_distributed_minibatch::update() {
  DataReader *data_reader = target_layer::select_data_reader();
  if(m_shared_data_reader) { /// If the data reader is shared with an input layer, don't update the reader
    return true;
  }else {
    return data_reader->update();
  }
}
