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
 protected:
  Mat X_local; /** Local matrix that holds data from data reader */
  Mat X_local_v; /** View of local matrix that holds data from data reader */
  CircMat Xs; /** Distributed matrix used to stage local data to layer output */

 public:
  input_layer_distributed_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers)
    : generic_data_distribution(comm, num_parallel_readers, data_readers),
      input_layer(comm, num_parallel_readers, data_readers),
      distributed_minibatch(comm, num_parallel_readers, data_readers),
      Xs(comm->get_model_grid()) {

    // Setup the data distribution
    initialize_distributed_matrices();
  }

  /** Returns description of ctor params */
  std::string get_description() const {
    return std::string {} + " input_layer_distributed_minibatch "
           + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  input_layer_distributed_minibatch(
    const input_layer_distributed_minibatch&) = default;
  input_layer_distributed_minibatch& operator=(
    const input_layer_distributed_minibatch&) = default;
  input_layer_distributed_minibatch* copy() const {
    return new input_layer_distributed_minibatch(*this);
  }

  std::string get_type() const { return "input:distributed"; }

  virtual inline void initialize_distributed_matrices() {
    input_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_data() {
    input_layer::setup_data();
    int max_mb_size = this->m_neural_network_model->get_max_mini_batch_size();
    if(io_layer::m_data_sets_span_models) {
      distributed_minibatch::calculate_num_iterations_per_epoch_training_spans_models(max_mb_size);
    } else {
      distributed_minibatch::calculate_num_iterations_per_epoch_training_unique_per_models(max_mb_size);
    }

    X_local.Resize(this->m_num_neurons, max_mb_size);

    distributed_minibatch::m_local_data_valid = false;
    distributed_minibatch::m_local_reader_done = false;
    distributed_minibatch::m_num_data_per_epoch = 0;
  }

 protected:
  void fp_set_std_matrix_view() {
    input_layer::fp_set_std_matrix_view();
    El::Int cur_mini_batch_size = m_neural_network_model->get_current_mini_batch_size();
    El::View(X_local_v, X_local, El::ALL, El::IR(0, cur_mini_batch_size));
  }

  /** Handle forward propagation (arguments are unused). */
  void fp_compute() {

    int num_samples_in_batch = distributed_minibatch::fetch_to_local_matrix(X_local_v);
    if(distributed_minibatch::is_current_root()) {
      /// Only update the number of samples processed by this parallel reader, when it is the current root
      input_layer::update_num_samples_processed(num_samples_in_batch);
    }

    /// Let each rank know this size of the current mini-batch
    /// Note that this field has to be updated before distributing the data
    this->m_neural_network_model->set_current_mini_batch_size(Layer::m_comm->model_broadcast(distributed_minibatch::m_root, num_samples_in_batch));

    distributed_minibatch::distribute_from_local_matrix(X_local, Xs);

    Copy(Xs, *this->m_activations);
  }

 public:
  /**
   * Once a mini-batch is processed, resuffle the data for the next batch if necessary
   */
  bool update_compute() {
    return distributed_minibatch::is_data_set_processed();
  }


  int fetch_from_data_reader(Mat& M_local) {
    generic_data_reader *data_reader = input_layer::select_data_reader();
    return data_reader->fetch_data(M_local);
  }

  void preprocess_data_samples(Mat& M_local, int num_samples_in_batch) {
    return;
  }

  bool update_data_reader(bool is_active_reader) {
    generic_data_reader *data_reader = input_layer::select_data_reader();
    return data_reader->update(is_active_reader);
  }

  execution_mode get_execution_mode() const {
    return this->m_execution_mode;
  }

  Mat *get_local_mat() {
    return &X_local;
  }

  CircMat *get_dist_mat() {
    return &Xs;
  }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_INPUT_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
