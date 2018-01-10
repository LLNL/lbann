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

#ifndef LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
#define LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED

#include "lbann/layers/io/target/generic_target_layer.hpp"
#include "lbann/data_distributions/distributed_io_buffer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
template <data_layout T_layout>
class target_layer_distributed_minibatch : public generic_target_layer {
 public:
  target_layer_distributed_minibatch(lbann_comm *comm, generic_input_layer *input_layer, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers, bool shared_data_reader, bool for_regression = false)
    : generic_target_layer(comm, input_layer, data_readers, for_regression){
    io_buffer = new distributed_io_buffer(comm, num_parallel_readers, data_readers);
    // Setup the data distribution
    initialize_distributed_matrices();

    // if(!dynamic_cast<input_layer_distributed_minibatch*>(input_layer)) {
    //   std::stringstream err;
    //   err << __FILE__ << " " << __LINE__
    //       << " :: " << get_type() << " paired with invalid input layer type" << std::endl;
    //   throw lbann_exception(err.str());
    // }
    io_buffer->fetch_data_fn = new fetch_data_functor(false, generic_target_layer::is_for_regression());
    io_buffer->update_data_reader_fn = new update_data_reader_functor(false);
  }
  target_layer_distributed_minibatch(
    const target_layer_distributed_minibatch&) = default;
  target_layer_distributed_minibatch& operator=(
    const target_layer_distributed_minibatch&) = default;
  target_layer_distributed_minibatch* copy() const override {
    return new target_layer_distributed_minibatch(*this);
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} + " target_layer_distributed_minibatch "
           + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  std::string get_type() const override { return "target:distributed"; }

  virtual inline void initialize_distributed_matrices() {
    generic_target_layer::initialize_distributed_matrices<T_layout>();
  }
  data_layout get_data_layout() const override { return T_layout; }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
