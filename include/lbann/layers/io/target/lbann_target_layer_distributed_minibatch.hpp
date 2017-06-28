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

#ifndef LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
#define LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED

#include "lbann/layers/io/target/lbann_target_layer.hpp"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
template <data_layout T_layout>
class target_layer_distributed_minibatch : public target_layer {
 protected:
  int m_root; /* Which rank is the root of the CircMat */
  Mat Y_local;
  CircMat Ys;

 public:
  target_layer_distributed_minibatch(lbann_comm *comm, uint mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers, bool shared_data_reader, bool for_regression = false)
    : target_layer(comm, mini_batch_size, data_readers, shared_data_reader, for_regression), Ys(this->m_comm->get_model_grid()) {
    // Setup the data distribution
    initialize_distributed_matrices();
    //  m_index = index;
    m_root = 0;
    //  m_num_neurons = m_training_data_reader->get_linearized_label_size(); /// @todo m_num_neurons should be hidden inside of an accessor function
  }

  std::string get_name() const { return "target layer distributed minibatch"; }

  virtual inline void initialize_distributed_matrices() {
    target_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual inline data_layout get_data_layout() { return T_layout; }

  void setup(int num_prev_neurons) {
    target_layer::setup(num_prev_neurons);
    if(!this->m_shared_data_reader) { /// If the target layer shares a data reader with an input layer, do not setup the data reader a second time
      if(io_layer::m_data_sets_span_models) {
        io_layer::setup_data_readers_for_training(0, Layer::m_comm->get_num_models() * Layer::m_mini_batch_size,
                                                            Layer::m_comm->get_model_rank() * Layer::m_mini_batch_size);
        io_layer::setup_data_readers_for_evaluation(0, this->m_mini_batch_size);
      } else {
        io_layer::setup_data_readers_for_training(0, this->m_mini_batch_size);
        io_layer::setup_data_readers_for_evaluation(0, this->m_mini_batch_size);
      }
    }

    /// @todo put in warning about bad target size
    if(static_cast<uint>(num_prev_neurons) != this->m_num_neurons) {
      throw -1;
    }

    Zeros(*this->m_error_signal, this->m_num_neurons, this->m_mini_batch_size);
    Zeros(Y_local, this->m_num_neurons, this->m_mini_batch_size);
    Zeros(Ys, this->m_num_neurons, this->m_mini_batch_size);

  }

  void fp_compute() {
    generic_data_reader *data_reader = target_layer::select_data_reader();

    if (this->m_comm->get_rank_in_model() == m_root) {
      Zero(Y_local);
      data_reader->fetch_label(Y_local);
    }

    if (this->m_comm->get_rank_in_model() == m_root) {
      CopyFromRoot(Y_local, Ys);
    } else {
      CopyFromNonRoot(Ys);
    }

    this->m_comm->model_barrier();
    Copy(Ys, *this->m_activations);

    /// Compute and record the objective function score
    DataType avg_error = this->m_neural_network_model->m_obj_fn->compute_obj_fn(*this->m_prev_activations_v, *this->m_activations_v);
    this->m_neural_network_model->m_obj_fn->record_obj_fn(this->m_execution_mode, avg_error);

    int64_t curr_mini_batch_size = this->m_neural_network_model->get_current_mini_batch_size();
    for (auto&& m : this->m_neural_network_model->m_metrics) {
      double cur_num_errors = (int) m->compute_metric(*this->m_prev_activations_v, *this->m_activations_v);
      m->record_error(cur_num_errors, curr_mini_batch_size);
    }

    return;
  }

  void bp_compute() {

    // Compute initial error signal
    // TODO: Replace with previous layer pointer.
    Layer* prev_layer = this->m_neural_network_model->get_layers()[this->m_index - 1];
    this->m_neural_network_model->m_obj_fn->compute_obj_fn_derivative(prev_layer,
                                                                      *this->m_prev_activations_v,
                                                                      *this->m_activations_v,
                                                                      *this->m_error_signal_v);

  }

  /**
   * Once a mini-batch is processed, resuffle the data for the next batch if necessary
   */
  bool update_compute() {
    generic_data_reader *data_reader = target_layer::select_data_reader();
    if(this->m_shared_data_reader) { /// If the data reader is shared with an input layer, don't update the reader
      return true;
    } else {
      return data_reader->update();
    }
  }
};
}

#endif  // LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
