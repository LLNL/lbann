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

#ifndef LBANN_LAYERS_REPEATED_INPUT_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_REPEATED_INPUT_LAYER_HPP_INCLUDED

#include "lbann/layers/io/input/input_layer.hpp"

namespace lbann {

/** Input layer for visual attention models.
 *  This layer outputs a data sample and label concatenated
 *  together. This is duplicated num_steps times.
 *  @todo This is very much a kludge. Deprecate as soon as possible.
 */
class repeated_input_layer : public input_layer<partitioned_io_buffer, data_layout::DATA_PARALLEL> {
 private:
  int m_num_steps;

  generic_io_buffer* m_label_io_buffer;

 public:

  repeated_input_layer(lbann_comm *comm,
                       int num_parallel_readers,
                       std::map<execution_mode, generic_data_reader *> data_readers,
                       int num_steps,
                       bool data_set_spans_models = true,
                       bool for_regression = false)
    : input_layer<partitioned_io_buffer, data_layout::DATA_PARALLEL>(comm,
                                                                     num_parallel_readers,
                                                                     data_readers,
                                                                     data_set_spans_models,
                                                                     for_regression),
      m_num_steps(num_steps) {

    m_label_io_buffer = new partitioned_io_buffer(comm,
                                                  num_parallel_readers,
                                                  data_readers);
    m_label_io_buffer->fetch_data_fn = new fetch_data_functor(false, false);
    m_label_io_buffer->update_data_reader_fn = new update_data_reader_functor(false);

  }

  ~repeated_input_layer() override {
    if (m_label_io_buffer != nullptr) { delete m_label_io_buffer; }
  }

  repeated_input_layer* copy() const override {
    return new repeated_input_layer(*this);
  }

  std::string get_type() const override {
    return std::string {}
      + "repeated_input:"
      + m_io_buffers[0]->get_type();
  }

  void setup_dims() override {
    input_layer<partitioned_io_buffer, data_layout::DATA_PARALLEL>::setup_dims();
    const auto& data_size = get_linearized_data_size();
    const auto& label_size = get_linearized_label_size();
    this->m_neuron_dims.assign(1, m_num_steps * (data_size + label_size));
    this->m_num_neuron_dims = this->m_neuron_dims.size();
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());
  }

  void setup_data() override {
    input_layer<partitioned_io_buffer, data_layout::DATA_PARALLEL>::setup_data();
    const auto& max_mb_size = this->m_model->get_max_mini_batch_size();
    const auto& data_size = get_linearized_data_size();
    const auto& label_size = get_linearized_label_size();
    for (auto& io_buffer : m_io_buffers) {
      io_buffer->setup_data(data_size, label_size, max_mb_size);
    }
  }

  void fp_compute() override {

    // Output matrix
    auto& local_output = get_local_activations();

    // Create staging area for data and labels
    const auto& local_width = local_output.Width();
    const auto& data_size = get_linearized_data_size();
    const auto& label_size = get_linearized_label_size();
    const auto& mode = this->m_model->get_execution_mode();
    const auto& mini_batch_size = this->m_model->get_current_mini_batch_size();
    CPUMat data(data_size, local_width), labels(label_size, local_width);
    for (auto& io_buffer : m_io_buffers) {
      io_buffer->set_local_matrix_bypass(&data, 0);
      io_buffer->fp_setup_data(mini_batch_size, 0);
      io_buffer->set_local_matrix_bypass(&labels, 1);
      io_buffer->fp_setup_data(mini_batch_size, 1);
    }

    /// support for data_store out-of-memory mode; this instructs
    /// the data_store (via the data_reader) to read in the
    /// next mb from file, then exchange data as needed
    get_data_reader()->init_minibatch();

    // Get data and labels
    m_io_buffers[0]->fetch_to_local_matrix(get_data_reader(), mode);

    // Copy data and labels into output
    CPUMat output_v;
    for (int i = 0; i < m_num_steps; ++i) {
      const auto& data_start = i * (data_size + label_size);
      const auto& data_end = data_start + data_size;
      const auto& label_start = data_end;
      const auto& label_end = label_start + label_size;
      El::View(output_v, local_output, El::IR(data_start, data_end), El::ALL);
      El::Copy(data, output_v);
      El::View(output_v, local_output, El::IR(label_start, label_end), El::ALL);
      El::Copy(labels, output_v);
    }

    // Use the predetermined size of the mini-batch to set the current
    // batch size for the neural network
    update_num_samples_processed(get_current_mini_batch_size());

  }


};

}

#endif  // LBANN_LAYERS_REPEATED_INPUT_LAYER_HPP_INCLUDED
