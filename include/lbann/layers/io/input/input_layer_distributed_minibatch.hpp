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
#include "lbann/data_distributions/distributed_io_buffer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
template <data_layout T_layout>
class input_layer_distributed_minibatch : public input_layer {
 public:
  input_layer_distributed_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers, bool data_set_spans_models = true)
    : input_layer(comm, num_parallel_readers, data_readers, data_set_spans_models) {
    io_buffer = new distributed_io_buffer(comm, num_parallel_readers, data_readers);
    io_buffer->fetch_data_fn = new fetch_data_functor(true, false);
    io_buffer->update_data_reader_fn = new update_data_reader_functor(true);
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

  // std::string get_type() const override { return "input:distributed"; }

  virtual inline void initialize_distributed_matrices() {
    input_layer::initialize_distributed_matrices<T_layout>();
  }
  data_layout get_data_layout() const override { return T_layout; }

  void setup_data() override {
    input_layer::setup_data();
  }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_INPUT_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
