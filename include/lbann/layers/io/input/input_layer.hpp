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

#ifndef LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED

#include "lbann/layers/io/io_layer.hpp"
#include "lbann/data_distributions/data_distribution.hpp"
#include "lbann/models/model.hpp"

namespace lbann {
class input_layer : public io_layer, public virtual generic_data_distribution {
 public:
  input_layer(lbann_comm *comm, int num_parallel_readers,  std::map<execution_mode, generic_data_reader *> data_readers)
    : generic_data_distribution(comm, num_parallel_readers, data_readers),
      io_layer(comm, data_readers) {}

  virtual ~input_layer() {
    // Input layer always frees data readers.
    if (m_training_dataset.m_data_reader != nullptr) {
      delete m_training_dataset.m_data_reader;
      m_training_dataset.m_data_reader = nullptr;
    }
    if (m_validation_dataset.m_data_reader != nullptr) {
      delete m_validation_dataset.m_data_reader;
      m_validation_dataset.m_data_reader = nullptr;
    }
    if (m_testing_dataset.m_data_reader != nullptr) {
      delete m_testing_dataset.m_data_reader;
      m_testing_dataset.m_data_reader = nullptr;
    }
  }

  // Input layers copy their datareaders.
  input_layer(const input_layer& other) : generic_data_distribution(other), io_layer(other) {
    if (m_training_dataset.m_data_reader) {
      m_training_dataset.m_data_reader = m_training_dataset.m_data_reader->copy();
    }
    if (m_validation_dataset.m_data_reader) {
      m_validation_dataset.m_data_reader = m_validation_dataset.m_data_reader->copy();
    }
    if (m_testing_dataset.m_data_reader) {
      m_testing_dataset.m_data_reader = m_testing_dataset.m_data_reader->copy();
    }
  }

  input_layer& operator=(const input_layer& other) {
    generic_data_distribution::operator=(other);
    io_layer::operator=(other);
    if (m_training_dataset.m_data_reader) {
      m_training_dataset.m_data_reader = m_training_dataset.m_data_reader->copy();
    }
    if (m_validation_dataset.m_data_reader) {
      m_validation_dataset.m_data_reader = m_validation_dataset.m_data_reader->copy();
    }
    if (m_testing_dataset.m_data_reader) {
      m_testing_dataset.m_data_reader = m_testing_dataset.m_data_reader->copy();
    }
    return *this;
  }

  void setup_dims() {
    io_layer::setup_dims();
    this->m_neuron_dims = io_layer::get_data_dims();
    this->m_num_neuron_dims = this->m_neuron_dims.size();
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());
  }

  void setup_data() {
    io_layer::setup_data();

    // in case that target_layer gets initialized beforehand
    if(m_training_dataset.m_data_reader != nullptr) {
      m_training_dataset.m_data_reader->setup();
    }
    if(m_validation_dataset.m_data_reader != nullptr) {
      m_validation_dataset.m_data_reader->setup();
    }
    if(m_testing_dataset.m_data_reader != nullptr) {
      m_testing_dataset.m_data_reader->setup();
    }
  }

  template<data_layout T_layout> inline void initialize_distributed_matrices() {
    io_layer::initialize_distributed_matrices<T_layout>();
  }

  /** Define the standard view of the matrix -- and set it for the model
   * Setup the effective (global) mini-batch size so that gradients are properly
   * averaged across models. */
  virtual void fp_set_std_matrix_view() {
    // Use the predetermined size of the mini-batch to set the current
    // batch size for the neural network
    El::Int cur_mini_batch_size = generic_data_distribution::get_current_mini_batch_size();
    this->m_neural_network_model->set_current_mini_batch_size(cur_mini_batch_size);

    // Use the precomputed size of the global mini-batch to set the
    // current effective batch size across all models
    int total_mini_batch_size = generic_data_distribution::get_current_global_mini_batch_size();
    this->m_neural_network_model->set_effective_mini_batch_size(total_mini_batch_size);

    // Once the current mini-batch size is defined, set the standard view for activations only
    El::View(*m_activations_v, *m_activations, El::ALL, El::IR(0, cur_mini_batch_size));
  }

  /** No setting the standard view of the matrix -- it defines the standard view */
  virtual void bp_set_std_matrix_view() {}

  // save state of IO to a checkpoint
  bool saveToCheckpointShared(persist& p) {
    // save state of data readers from input layer
    this->m_training_dataset.m_data_reader->saveToCheckpointShared(p, "data_reader_training");
    this->m_validation_dataset.m_data_reader->saveToCheckpointShared(p, "data_reader_validation");
    this->m_testing_dataset.m_data_reader->saveToCheckpointShared(p, "data_reader_testing");

    // save our own state
    io_layer::saveToCheckpointShared(p);

    return true;
  }

  // reload state of IO from a checkpoint
  bool loadFromCheckpointShared(persist& p) {
    // save state of data readers from input layer
    this->m_training_dataset.m_data_reader->loadFromCheckpointShared(p, "data_reader_training");
    this->m_validation_dataset.m_data_reader->loadFromCheckpointShared(p, "data_reader_validation");
    this->m_testing_dataset.m_data_reader->loadFromCheckpointShared(p, "data_reader_testing");

    // save our own state
    io_layer::loadFromCheckpointShared(p);

    return true;
  }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
