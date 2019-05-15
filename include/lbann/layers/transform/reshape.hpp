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

#ifndef RESHAPE_HPP_INCLUDED
#define RESHAPE_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** @brief Reshape tensor.
 *
 *  Forward and backward prop simply involve setting up tensor views,
 *  and hence are very cheap.
 */
template <data_layout T_layout, El::Device Dev>
class reshape_layer : public transform_layer {
public:
  reshape_layer(lbann_comm *comm,
                std::vector<int> dims)
    : transform_layer(comm) {
    set_output_dims(dims);
  }
  reshape_layer* copy() const override { return new reshape_layer(*this); }
  std::string get_type() const override { return "reshape"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:

  void setup_dims() override {
    transform_layer::setup_dims();

    const auto& input_dims = get_input_dims();
    auto output_dims = get_output_dims();

    // Determine any unspecified dimensions
    int unspecified_dim = -1;
    for (size_t dim = 0; dim < output_dims.size(); ++dim) {
      if (output_dims[dim] <= 0) {
        if (unspecified_dim < 0) { unspecified_dim = dim; }
        output_dims[dim] = 1;
      }
    }
    if (unspecified_dim >= 0) {
      const auto& specified_size = std::accumulate(output_dims.begin(),
                                                   output_dims.end(),
                                                   1,
                                                   std::multiplies<int>());
      output_dims[unspecified_dim] = get_input_size() / specified_size;
      set_output_dims(output_dims);
    }

    // Check that reshape is valid
    if (get_input_size() != get_output_size()) {
      std::stringstream err;
      err << "input tensor dimensions (";
      for (size_t i = 0; i < input_dims.size(); ++i) {
        err << (i > 0 ? " x " : "") << input_dims[i];
      }
      err << ") do not match output tensor dimensions (";
      for (size_t i = 0; i < output_dims.size(); ++i) {
        err << (i > 0 ? "x" : "") << output_dims[i];
      }
      err << ")";
      LBANN_ERROR(err.str());
    }

  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    El::LockedView(get_activations(), get_prev_activations());
  }
  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override {
    El::LockedView(get_error_signals(), get_prev_error_signals());
  }
  void fp_compute() override {}
  void bp_compute() override {}

};

} // namespace lbann

#endif // RESHAPE_HPP_INCLUDED
