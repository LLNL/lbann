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
  CircMat Xs; /** Distributed matrix used to stage local data to layer output */

 public:
  input_layer_distributed_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers)
    : input_layer(comm, num_parallel_readers, data_readers),
      distributed_minibatch(comm, num_parallel_readers, data_readers),
      Xs(comm->get_model_grid()) {

    // Setup the data distribution
    initialize_distributed_matrices();
  }
  input_layer_distributed_minibatch(
    const input_layer_distributed_minibatch&) = default;
  input_layer_distributed_minibatch& operator=(
    const input_layer_distributed_minibatch&) = default;
  input_layer_distributed_minibatch* copy() const {
    return new input_layer_distributed_minibatch(*this);
  }

  std::string get_name() const { return "input layer distributed minibatch parallel io"; }

  virtual inline void initialize_distributed_matrices() {
    input_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_data() {
    input_layer::setup_data();
    int max_mb_size = this->m_neural_network_model->get_max_mini_batch_size();
    if(io_layer::m_data_sets_span_models) {
      int stride = Layer::m_comm->get_num_models() * distributed_minibatch::m_num_parallel_readers_training * max_mb_size;
      int base_offset = Layer::m_comm->get_rank_in_model() * Layer::m_comm->get_num_models() * max_mb_size;
      int model_offset = Layer::m_comm->get_model_rank() * max_mb_size;
      //cout << "["<< Layer::m_comm->get_rank_in_world() << "] Setting up input layer, with " << Layer::m_comm->get_num_models() << " models and " << m_num_parallel_readers_training << " parallel readers and " << Layer::m_mini_batch_size << " mb size, which gives a stride of " << stride << " and my model offset is " << model_offset << " and my base offset is " << base_offset /*(Layer::m_comm->get_rank_in_model() * Layer::m_mini_batch_size)*/ << endl;
      io_layer::setup_data_readers_for_training(base_offset,
                                                          stride, 1,
                                                          model_offset);
      distributed_minibatch::calculate_num_iterations_per_epoch_spanning_models(max_mb_size,
                                                                this->m_training_dataset.data_reader);
      /// Note that the data readers for evaluation should not be partitioned over multiple models (otherwise each model will be scored on a different set of data)
      io_layer::setup_data_readers_for_evaluation(Layer::m_comm->get_rank_in_model() * max_mb_size,
                                                            distributed_minibatch::m_num_parallel_readers_training * max_mb_size);
      distributed_minibatch::calculate_num_iterations_per_epoch_single_model(max_mb_size,
                                                                this->m_validation_dataset.data_reader);
      distributed_minibatch::calculate_num_iterations_per_epoch_single_model(max_mb_size,
                                                                this->m_testing_dataset.data_reader);
    } else {
      io_layer::setup_data_readers_for_training(Layer::m_comm->get_rank_in_model() * max_mb_size,
                                                          distributed_minibatch::m_num_parallel_readers_training * max_mb_size);
      io_layer::setup_data_readers_for_evaluation(Layer::m_comm->get_rank_in_model() * max_mb_size,
                                                            distributed_minibatch::m_num_parallel_readers_training * max_mb_size);
    }

    X_local.Resize(this->m_num_neurons, max_mb_size);

    distributed_minibatch::m_local_data_valid = false;
    distributed_minibatch::m_local_reader_done = false;
    distributed_minibatch::m_num_data_per_epoch = 0;
  }

 protected:
  /** Handle forward propagation (arguments are unused). */
  void fp_compute() {

    int num_samples_in_batch = distributed_minibatch::fetch_to_local_matrix(X_local);
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

  bool update_data_reader() {
    generic_data_reader *data_reader = input_layer::select_data_reader();
    return data_reader->update();
  }

  execution_mode get_execution_mode() {
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
