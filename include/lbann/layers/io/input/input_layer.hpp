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

namespace lbann {
class input_layer : public io_layer {
 public:
  input_layer(lbann_comm *comm, int mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers)
    : io_layer(comm, mini_batch_size, data_readers) {}

  void setup_dims() {
    io_layer::setup_dims();
    this->m_neuron_dims = io_layer::get_data_dims();
    this->m_num_neuron_dims = this->m_neuron_dims.size();
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());
  }

  template<data_layout T_layout> inline void initialize_distributed_matrices() {
    io_layer::initialize_distributed_matrices<T_layout>();
  }

  /** No setting the standard view of the matrix -- it defines the standard view */
  void fp_set_std_matrix_view(void) {}
  void bp_set_std_matrix_view(void) {}

  // save state of IO to a checkpoint
  bool saveToCheckpointShared(persist& p) {
    // save state of data readers from input layer
    this->m_training_dataset.data_reader->saveToCheckpointShared(p, "data_reader_training");
    this->m_validation_dataset.data_reader->saveToCheckpointShared(p, "data_reader_validation");
    this->m_testing_dataset.data_reader->saveToCheckpointShared(p, "data_reader_testing");

    // save our own state
    io_layer::saveToCheckpointShared(p);

    return true;
  }

  // reload state of IO from a checkpoint
  bool loadFromCheckpointShared(persist& p) {
    // save state of data readers from input layer
    this->m_training_dataset.data_reader->loadFromCheckpointShared(p, "data_reader_training");
    this->m_validation_dataset.data_reader->loadFromCheckpointShared(p, "data_reader_validation");
    this->m_testing_dataset.data_reader->loadFromCheckpointShared(p, "data_reader_testing");

    // save our own state
    io_layer::loadFromCheckpointShared(p);

    return true;
  }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
