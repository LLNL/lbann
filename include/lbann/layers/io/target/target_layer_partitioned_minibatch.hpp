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

#ifndef LBANN_LAYERS_TARGET_LAYER_PARTITIONED_MINIBATCH_HPP_INCLUDED
#define LBANN_LAYERS_TARGET_LAYER_PARTITIONED_MINIBATCH_HPP_INCLUDED

#include "lbann/layers/io/target/target_layer.hpp"
#include "lbann/data_distributions/partitioned_minibatch.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {

template <data_layout T_layout = data_layout::DATA_PARALLEL>
class target_layer_partitioned_minibatch : public target_layer, public partitioned_minibatch {
 public:
  target_layer_partitioned_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers, bool shared_data_reader, bool for_regression=false)
    : generic_data_distribution(comm, num_parallel_readers, data_readers),
      target_layer(comm, data_readers, shared_data_reader, for_regression),
      partitioned_minibatch(comm, std::min(num_parallel_readers, Layer::m_comm->get_procs_per_model()), data_readers) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "partitioned_minibatch only supports DATA_PARALLEL");
    // Setup the data distribution
    initialize_distributed_matrices();
  }

  target_layer_partitioned_minibatch* copy() const {
    throw lbann_exception("target_layer_partitioned_minibatch can't be copied");
    return nullptr;
  }

  std::string get_name() const { return "target:partitioned"; }

  virtual inline void initialize_distributed_matrices() {
    target_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  virtual void setup_data() {
    target_layer::setup_data();

    int max_mb_size = this->m_neural_network_model->get_max_mini_batch_size();
    if(!this->m_shared_data_reader) { /// If the target layer shares a data reader with an input layer, do not setup the data reader a second time
      if(io_layer::m_data_sets_span_models) {
        int base_offset = Layer::m_comm->get_rank_in_model();
        int batch_stride = Layer::m_comm->get_num_models() * max_mb_size;
        int model_offset = Layer::m_comm->get_model_rank() * max_mb_size;
        cout << "Setting up target layer, with " << Layer::m_comm->get_num_models() << " models and " << m_num_parallel_readers_training << " parallel readers and " << max_mb_size << " mb size, which gives a stride of " << batch_stride << endl;
        io_layer::setup_data_readers_for_training(base_offset,
                                                  batch_stride,
                                                  m_num_parallel_readers_training,
                                                  model_offset);
        partitioned_minibatch::calculate_num_iterations_per_epoch_spanning_models(max_mb_size,
                                                                                  this->m_training_dataset.data_reader);
        /// Note that the data readers for evaluation should not be partitioned over multiple models (otherwise each model will be scored on a different set of data)
        io_layer::setup_data_readers_for_evaluation(Layer::m_comm->get_rank_in_model(),
                                                    max_mb_size,
                                                    m_num_parallel_readers_testing);
        partitioned_minibatch::calculate_num_iterations_per_epoch_single_model(max_mb_size,
                                                                               this->m_validation_dataset.data_reader);
        partitioned_minibatch::calculate_num_iterations_per_epoch_single_model(max_mb_size, 
                                                                               this->m_testing_dataset.data_reader);
      } else {
        io_layer::setup_data_readers_for_training(Layer::m_comm->get_rank_in_model(),
                                                  max_mb_size,
                                                  m_num_parallel_readers_training);
        io_layer::setup_data_readers_for_evaluation(Layer::m_comm->get_rank_in_model(),
                                                    max_mb_size,
                                                    m_num_parallel_readers_testing);
      }
    }

    m_local_data_valid = false;
    m_local_reader_done = false;
    m_num_data_per_epoch = 0;
  }

  void fp_compute() {
    int num_samples_in_batch = fetch_to_local_matrix(this->m_activations_v->Matrix());

    target_layer::update_num_samples_processed(num_samples_in_batch);

    int curr_mini_batch_size = this->m_neural_network_model->get_current_mini_batch_size();

    /// Compute and record the objective function score
    DataType avg_error = this->m_neural_network_model->m_obj_fn->compute_obj_fn(*this->m_prev_activations_v, *this->m_activations_v);
    this->m_neural_network_model->m_obj_fn->record_obj_fn(this->m_execution_mode, avg_error);

    for (auto&& m : this->m_neural_network_model->get_metrics()) {
      double num_errors = m->compute_metric(*this->m_prev_activations_v, *this->m_activations_v);
      m->record_error(num_errors, curr_mini_batch_size);
    }

    return;
  }


  void bp_compute() {

    // Compute initial error signal
    this->m_neural_network_model->m_obj_fn->compute_obj_fn_derivative(*m_prev_layer,
                                                                      *this->m_prev_activations_v,
                                                                      *this->m_activations_v,
                                                                      *this->m_error_signal_v);
  }

  /**
   * Once a mini-batch is processed, resuffle the data for the next batch if necessary
   */
  bool update_compute() {
    return is_data_set_processed();
  }

  int fetch_from_data_reader(Mat& M_local) {
    generic_data_reader *data_reader = target_layer::select_data_reader();
    if (target_layer::is_for_regression()) {
      return data_reader->fetch_responses(M_local);
    } else {
      return data_reader->fetch_labels(M_local);
    }
  }

  void preprocess_data_samples(Mat& M_local, int num_samples_in_batch) {
    return;
  }

  bool update_data_reader() {
    generic_data_reader *data_reader = target_layer::select_data_reader();
    if(this->m_shared_data_reader) {
      /// If the data reader is shared with an input layer, don't update the reader just check to see if the epoch is done
      /// or will be done on the next update of the input layer (which includes adding the stride).
      /// Note that target layers are always update before input layers, which is why the position
      /// is not up to date yet.
      return (data_reader->get_next_position() < data_reader->get_num_data());
    } else {
      return data_reader->update();
    }
  }

  execution_mode get_execution_mode() {
    return this->m_execution_mode;
  }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_TARGET_LAYER_PARTITIONED_MINIBATCH_HPP_INCLUDED
