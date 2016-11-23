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

lbann::target_layer_distributed_minibatch::target_layer_distributed_minibatch(lbann_comm* comm, uint mini_batch_size, std::map<execution_mode, DataReader*> data_readers, bool shared_data_reader)
  : target_layer(comm, mini_batch_size, data_readers, shared_data_reader), Ys(comm->get_model_grid())
{
  //  Index = index;
  m_root = 0;
  //  NumNeurons = m_training_data_reader->get_linearized_label_size(); /// @todo NumNeurons should be hidden inside of an accessor function
}

void lbann::target_layer_distributed_minibatch::setup(int num_prev_neurons) {
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

///@todo update this to use the new fp_linearity framework
DataType lbann::target_layer_distributed_minibatch::forwardProp(DataType prev_WBL2NormSum) {
  DataReader *data_reader = target_layer::select_data_reader();

  *m_prev_activations = *fp_input; // BVE this should be handled in the new fp framework

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

  /// Check to see if the predicted results match the target results
  int num_errors = 0;
  /// @todo this needs to be optimized so that it is done locally
  /// first then aggregated
  for (int n = 0; n < Y_local.Width(); n++) {
    int targetidx = -1;
    float targetmax = 0;
    for (int m = 0; m < Y_local.Height(); m++) {
      if (targetmax < Y_local.Get(m, n)) {
        targetmax = Y_local.Get(m, n);
        targetidx = m;
      }
    }
    int labelidx = -1;
    float labelmax = 0;
    for (int m = 0; m < fp_input->Height(); m++) {
      if (labelmax < fp_input->Get(m, n)) {
        labelmax = fp_input->Get(m, n);
        labelidx = m;
      }
    }
    if (targetidx != labelidx)
      num_errors++;
      
  }
 
  return num_errors;
}

void lbann::target_layer_distributed_minibatch::backProp() {
  /// Compute the error between the target values and the previous layer's activations
  /// Copy the results to the m_error_signal variable for access by the next lower layer
  Copy(*m_prev_activations, *m_error_signal); // delta = (activation - y)
  Axpy(-1., Ys, *m_error_signal); // Per-neuron error
  Copy(Ys, *m_activations);

  if (m_execution_mode == execution_mode::training) {
    DataType avg_error = neural_network_model->obj_fn->compute_obj_fn(*m_prev_activations_v, *m_activations_v);
    aggregate_cost += avg_error;
    num_backprop_steps++;
  }
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
