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

#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/data_distributions/distributed_minibatch.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
template <data_layout T_layout>
class input_layer_distributed_minibatch : public input_layer, public distributed_minibatch {
 public:
  input_layer_distributed_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers, bool data_set_spans_models = true)
    : generic_data_distribution(comm, num_parallel_readers, data_readers),
      input_layer(comm, num_parallel_readers, data_readers, data_set_spans_models),
      distributed_minibatch(comm, num_parallel_readers, data_readers) {

    // Setup the data distribution
    initialize_distributed_matrices();
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} + " input_layer_distributed_minibatch "
           + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  input_layer_distributed_minibatch(
    const input_layer_distributed_minibatch&) = default;
  input_layer_distributed_minibatch& operator=(
    const input_layer_distributed_minibatch&) = default;
  input_layer_distributed_minibatch* copy() const override {
    return new input_layer_distributed_minibatch(*this);
  }

  std::string get_type() const override { return "input:distributed"; }

  virtual inline void initialize_distributed_matrices() {
    input_layer::initialize_distributed_matrices<T_layout>();
  }
  data_layout get_data_layout() const override { return T_layout; }

  void setup_data() override {
    input_layer::setup_data();
    int max_mb_size = this->m_neural_network_model->get_max_mini_batch_size();
    #ifdef LBANN_DEBUG
    std::cout << "Setting up data for the input layer " << io_layer::m_data_set_spans_models << std::endl;
    #endif
    if(io_layer::m_data_set_spans_models) {
      calculate_num_iterations_per_epoch_training_spans_models(max_mb_size);
    } else {
      calculate_num_iterations_per_epoch_training_unique_per_models(max_mb_size);
    }

    for (auto& buf : m_data_buffers) {
      buf.second->M_local.Resize(this->m_num_neurons, max_mb_size);
      buf.second->Ms.Resize(this->m_num_neurons, max_mb_size);
    }
  }

 protected:
  void fp_set_std_matrix_view() override {
    input_layer::fp_set_std_matrix_view();
    El::Int cur_mini_batch_size = m_neural_network_model->get_current_mini_batch_size();
    data_buffer *buf = distributed_minibatch::get_data_buffer();
    El::View(buf->M_local_v, buf->M_local, El::ALL, El::IR(0, cur_mini_batch_size));
  }

  /** Handle forward propagation (arguments are unused). */
  void fp_compute() override {
    data_buffer *buf = distributed_minibatch::get_data_buffer();
    int num_samples_in_batch = distributed_minibatch::fetch_to_local_matrix(buf->M_local_v, get_data_reader());
    if(distributed_minibatch::is_current_root()) {
      /// Only update the number of samples processed by this parallel reader, when it is the current root
      input_layer::update_num_samples_processed(num_samples_in_batch);
    }

    /// Let each rank know this size of the current mini-batch
    /// Note that this field has to be updated before distributing the data
    this->m_neural_network_model->set_current_mini_batch_size(Layer::m_comm->model_broadcast(distributed_minibatch::current_root_rank(), num_samples_in_batch));

    distributed_minibatch::distribute_from_local_matrix(buf->M_local, buf->Ms, get_data_reader());

    Copy(buf->Ms, *this->m_activations);
  }

 public:
  /**
   * Once a mini-batch is processed, resuffle the data for the next batch if necessary
   */
  bool update_compute() override {
    return distributed_minibatch::is_data_set_processed(get_data_reader());
  }

  void preprocess_data_samples(Mat& M_local, int num_samples_in_batch) override {
    return;
  }

  data_buffer *get_data_buffer() const {
    return distributed_minibatch::get_data_buffer(get_execution_mode());
  }

  Mat *get_local_mat() {
    data_buffer *buf = distributed_minibatch::get_data_buffer();
    return &buf->M_local;
  }

  CircMat *get_dist_mat() {
    data_buffer *buf = distributed_minibatch::get_data_buffer();
    return &buf->Ms;
  }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_INPUT_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
