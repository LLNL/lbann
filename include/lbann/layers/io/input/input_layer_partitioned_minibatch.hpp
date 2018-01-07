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

#ifndef LBANN_LAYERS_INPUT_LAYER_PARTITIONED_MINIBATCH_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_PARTITIONED_MINIBATCH_HPP_INCLUDED

#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/data_distributions/partitioned_io_buffer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {

template <data_layout T_layout = data_layout::DATA_PARALLEL>
class input_layer_partitioned_minibatch : public input_layer {
 public:
  /// @todo make the map and vector references
  input_layer_partitioned_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers, bool data_set_spans_models = true)
    : input_layer(comm, num_parallel_readers, data_readers, data_set_spans_models) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "partitioned_minibatch only supports DATA_PARALLEL");

    io_buffer = new partitioned_io_buffer(comm, std::min(num_parallel_readers, Layer::m_comm->get_procs_per_model()), data_readers);
    io_buffer->fetch_data_fn = new fetch_data_functor(true, false);
    io_buffer->update_data_reader_fn = new update_data_reader_functor(true);
    // Setup the data distribution
    initialize_distributed_matrices();
  }
  input_layer_partitioned_minibatch* copy() const override {
    return new input_layer_partitioned_minibatch(*this);
  }

  // std::string get_type() const override { return "input:partitioned"; }

  // /** Returns description of ctor params */
  // std::string get_description() const override {
  //   std::string s = get_topo_description();
  //   return std::string {} + " input_layer_partitioned_minibatch "
  //          + " dataLayout: " + this->get_data_layout_string(get_data_layout())
  //          + " (" + s + ")";
  // }

  virtual inline void initialize_distributed_matrices() {
    input_layer::initialize_distributed_matrices<T_layout>();
  }
  data_layout get_data_layout() const override { return T_layout; }

  void setup_data() override {
    input_layer::setup_data();
  }

  void fp_compute() override {
    io_buffer->fetch_to_local_matrix(get_data_reader(), this->m_model->get_execution_mode());

    // Use the predetermined size of the mini-batch to set the current
    // batch size for the neural network
    int num_samples_in_batch = get_current_mini_batch_size();

    input_layer::update_num_samples_processed(num_samples_in_batch);
  }
};

}

#endif  // LBANN_LAYERS_INPUT_LAYER_PARTITIONED_MINIBATCH_HPP_INCLUDED
