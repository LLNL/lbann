////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYER_SPLIT_HPP_INCLUDED
#define LBANN_LAYER_SPLIT_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Present input tensor to multiple outputs. */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class split_layer : public transform_layer {
public:

  split_layer(lbann_comm *comm) : transform_layer(comm) {
    this->m_expected_num_child_layers = -1; // No limit on children
  }

  split_layer* copy() const override { return new split_layer(*this); }
  std::string get_type() const override { return "split"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:

  void setup_dims() override {
    Layer::setup_dims();
    for (int i = 0; i < get_num_children(); ++i) {
      set_output_dims(get_input_dims(), i);
    }
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    const auto& input = get_prev_activations();
    for (int i = 0; i < get_num_children(); ++i) {
      El::LockedView(get_activations(i), input);
    }
  }

  void fp_compute() override {}

  void bp_compute() override {
    auto& gradient_wrt_input = get_error_signals();
    if (get_num_children() > 0) {
      El::Copy(get_prev_error_signals(0), gradient_wrt_input);
    } else {
      El::Zero(gradient_wrt_input);
    }
    for (int i = 1; i < get_num_children(); ++i) {
      El::Axpy(DataType(1), get_prev_error_signals(i),
               gradient_wrt_input);
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_SPLIT_HPP_INCLUDED
