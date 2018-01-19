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

#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {

template <typename T_io_buffer, data_layout T_layout = data_layout::DATA_PARALLEL>
class input_layer : public generic_input_layer {
 public:
  /// @todo make the map and vector references
  input_layer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers, bool data_set_spans_models = true)
    : generic_input_layer(comm, num_parallel_readers, data_readers, data_set_spans_models) {
    validate_data_layout();
    initialize_io_buffer(comm, std::min(num_parallel_readers, Layer::m_comm->get_procs_per_model()), data_readers);
    io_buffer->fetch_data_fn = new fetch_data_functor(true, false);
    io_buffer->update_data_reader_fn = new update_data_reader_functor(true);
    // Setup the data distribution
    initialize_distributed_matrices();
  }
  input_layer(const input_layer&) = default;
  input_layer& operator=(const input_layer&) = default;
  input_layer* copy() const override {
    return new input_layer(*this);
  }

  std::string get_type() const override {
    return std::string {}
      + "input:"
      + io_buffer->get_type();
  }

  inline void validate_data_layout();

  inline void initialize_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers) {
    generic_input_layer::initialize_io_buffer<T_io_buffer>(comm, num_parallel_readers, data_readers);
  }

  virtual inline void initialize_distributed_matrices() {
    generic_input_layer::initialize_distributed_matrices<T_layout>();
  }
  data_layout get_data_layout() const override { return T_layout; }

  void setup_data() override {
    generic_input_layer::setup_data();
  }
};

template<>
inline void input_layer<partitioned_io_buffer, data_layout::MODEL_PARALLEL>::validate_data_layout() {
  static_assert(true, "input_layer with partitioned_io_buffer does not supports MODEL_PARALLEL data layout");
}

template<>
inline void input_layer<partitioned_io_buffer, data_layout::DATA_PARALLEL>::validate_data_layout() {}

template<>
inline void input_layer<distributed_io_buffer, data_layout::MODEL_PARALLEL>::validate_data_layout() {}

template<>
inline void input_layer<distributed_io_buffer, data_layout::DATA_PARALLEL>::validate_data_layout() {}

}

#endif  // LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
