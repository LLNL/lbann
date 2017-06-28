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

#ifndef LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_PARALLEL_IO_HPP_INCLUDED
#define LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_PARALLEL_IO_HPP_INCLUDED

#include "lbann/layers/io/target/lbann_target_layer.hpp"
#include "lbann/io/lbann_distributed_minibatch_parallel_io.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
template <data_layout T_layout>
class target_layer_distributed_minibatch_parallel_io : public target_layer, public distributed_minibatch_parallel_io {
 protected:
  Mat Y_local;
  CircMat Ys;

 public:
  target_layer_distributed_minibatch_parallel_io(lbann_comm *comm, int num_parallel_readers, int mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers, bool shared_data_reader, bool for_regression = false)
    : target_layer(comm, mini_batch_size, data_readers, shared_data_reader, for_regression),
      distributed_minibatch_parallel_io(comm, num_parallel_readers, mini_batch_size, data_readers),
      Ys(comm->get_model_grid()) {
    // Setup the data distribution
    initialize_distributed_matrices();
  }

  std::string get_name() const { return "target layer distributed minibatch parallel io"; }

  virtual inline void initialize_distributed_matrices() {
    target_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual inline data_layout get_data_layout() { return T_layout; }

  virtual void setup(Layer *prev_layer, Layer *next_layer) {
    target_layer::setup(prev_layer, next_layer);

    if(!this->m_shared_data_reader) { /// If the target layer shares a data reader with an input layer, do not setup the data reader a second time
      if(io_layer::m_data_sets_span_models) {
        int stride = Layer::m_comm->get_num_models() * m_num_parallel_readers_training * Layer::m_mini_batch_size;
        int base_offset = Layer::m_comm->get_rank_in_model() * Layer::m_comm->get_num_models() * Layer::m_mini_batch_size;
        int model_offset = Layer::m_comm->get_model_rank() * Layer::m_mini_batch_size;
        //cout << "Setting up input layer, with " << Layer::m_comm->get_num_models() << " models and " << m_num_parallel_readers_training << " parallel readers and " << Layer::m_mini_batch_size << " mb size, which gives a stride of " << stride << endl;
        io_layer::setup_data_readers_for_training(base_offset,
                                                            stride,
                                                            model_offset);
        distributed_minibatch_parallel_io::calculate_num_iterations_per_epoch(this->m_training_dataset.data_reader);
        /// Note that the data readers for evaluation should not be partitioned over multiple models (otherwise each model will be scored on a different set of data)
        io_layer::setup_data_readers_for_evaluation(Layer::m_comm->get_rank_in_model() * Layer::m_mini_batch_size,
                                                              m_num_parallel_readers_training * Layer::m_mini_batch_size);
      } else {
        io_layer::setup_data_readers_for_training(Layer::m_comm->get_rank_in_model() * Layer::m_mini_batch_size,
                                                            m_num_parallel_readers_training * Layer::m_mini_batch_size);
        io_layer::setup_data_readers_for_evaluation(Layer::m_comm->get_rank_in_model() * Layer::m_mini_batch_size,
                                                              m_num_parallel_readers_training * Layer::m_mini_batch_size);
      }
    }

    Zeros(*this->m_error_signal, this->m_num_neurons, Layer::m_mini_batch_size);
    Zeros(Y_local, this->m_num_neurons, Layer::m_mini_batch_size);
    Zeros(Ys, this->m_num_neurons, Layer::m_mini_batch_size);
    Zeros(*this->m_prev_activations, this->m_num_prev_neurons, this->m_mini_batch_size);
    Zeros(*this->m_activations, this->m_num_neurons, this->m_mini_batch_size);

    m_local_data_valid = false;
    m_local_reader_done = false;
    m_num_data_per_epoch = 0;
  }

  void fp_compute() {
    int num_samples_in_batch = fetch_to_local_matrix(Y_local);
    if(is_current_root()) {
      /// Only update the number of samples processed by this parallel reader, when it is the current root
      target_layer::update_num_samples_processed(num_samples_in_batch);
    }

    int curr_mini_batch_size = this->m_neural_network_model->get_current_mini_batch_size();
    if(is_current_root() && num_samples_in_batch != curr_mini_batch_size) {
      throw lbann_exception("lbann_target_layer_distributed_minibatch_parallel_io: number of labels does not match the current mini-batch size.");
    }
    /// @todo should this distribute the entire matrix even if there is only a partial mini-batch
    distribute_from_local_matrix(Y_local, Ys);
    Copy(Ys, *this->m_activations);

    /// Compute and record the objective function score
    DataType avg_error = this->m_neural_network_model->m_obj_fn->compute_obj_fn(*this->m_prev_activations_v, *this->m_activations_v);
    this->m_neural_network_model->m_obj_fn->record_obj_fn(this->m_execution_mode, avg_error);

    for (auto&& m : this->m_neural_network_model->m_metrics) {
      double num_errors = m->compute_metric(*this->m_prev_activations_v, *this->m_activations_v);
      m->record_error(num_errors, curr_mini_batch_size);
    }

    return;
  }


  void bp_compute() {

    // Compute initial error signal
    this->m_neural_network_model->m_obj_fn->compute_obj_fn_derivative(m_prev_layer,
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
      return data_reader->fetch_response(M_local);
    } else {
      return data_reader->fetch_label(M_local);
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
      return (data_reader->get_next_position() < data_reader->getNumData());
    } else {
      return data_reader->update();
    }
  }

  execution_mode get_execution_mode() {
    return this->m_execution_mode;
  }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_PARALLEL_IO_HPP_INCLUDED
