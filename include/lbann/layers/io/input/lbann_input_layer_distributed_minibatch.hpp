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

#ifndef LBANN_LAYERS_INPUT_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED

#include "lbann/layers/io/input/lbann_input_layer.hpp"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
template <class T_layout>
class input_layer_distributed_minibatch : public input_layer<T_layout> {
 public:
  int m_root; /* Which rank is the root of the CircMat */
  Mat X_local;
  CircMat Xs;
  long m_num_data_per_epoch;

 public:
  input_layer_distributed_minibatch(T_layout data_dist, lbann_comm *comm, uint mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers, std::vector<regularizer *> regs = {})
    : input_layer<T_layout>(data_dist, comm, mini_batch_size, data_readers, regs), Xs(this->m_comm->get_model_grid()) {

    this->m_type = layer_type::input_distributed_minibatch;

    //  m_index = index;
    this->m_root = 0;
    this->m_num_data_per_epoch = 0;

  }

  void setup(int num_prev_neurons) {
    if(io_layer<T_layout>::m_data_sets_span_models) {
      io_layer<T_layout>::setup_data_readers_for_training(0, Layer::m_comm->get_num_models() * Layer::m_mini_batch_size,
                                                          Layer::m_comm->get_model_rank() * Layer::m_mini_batch_size);
      io_layer<T_layout>::setup_data_readers_for_evaluation(0, this->m_mini_batch_size);
    } else {
      io_layer<T_layout>::setup_data_readers_for_training(0, this->m_mini_batch_size);
      io_layer<T_layout>::setup_data_readers_for_evaluation(0, this->m_mini_batch_size);
    }

    Zeros(*this->m_activations, this->m_num_neurons, this->m_mini_batch_size);
    Zeros(X_local, this->m_num_neurons, this->m_mini_batch_size);
  }

 protected:
  /** Handle forward propagation (arguments are unused.) */
  void fp_linearity() {
    generic_data_reader *data_reader = input_layer<T_layout>::select_data_reader();
    int num_samples_in_batch = 0;

    if (this->m_comm->get_rank_in_model() == m_root) {
      Zero(X_local);
      num_samples_in_batch = data_reader->fetch_data(X_local);
      bool data_valid = (num_samples_in_batch > 0);
      if(data_valid) {
        m_num_data_per_epoch+=num_samples_in_batch;
      }
    }

    /// Let each rank know this size of the current mini-batch
    /// Note that this field has to be updated before distributing the data
    this->m_neural_network_model->set_current_mini_batch_size(Layer::m_comm->model_broadcast(m_root, num_samples_in_batch));

    if (this->m_comm->get_rank_in_model() == m_root) {
      CopyFromRoot(X_local, Xs);
    } else {
      CopyFromNonRoot(Xs);
    }

    this->m_comm->model_barrier();

    Copy(Xs, *this->m_activations);
  }

 public:
  /**
   * Once a mini-batch is processed, resuffle the data for the next batch if necessary
   */
  bool update() {
    generic_data_reader *data_reader = input_layer<T_layout>::select_data_reader();
    return !data_reader->update();
  }

  Mat *get_local_mat(void) {
    return &X_local;
  };
  CircMat *get_dist_mat(void) {
    return &Xs;
  };
};
}

#endif  // LBANN_LAYERS_INPUT_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
