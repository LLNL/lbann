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

#ifndef LBANN_LAYER_REDUCTION_HPP_INCLUDED
#define LBANN_LAYER_REDUCTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"

namespace lbann {

enum class reduction_mode {INVALID, SUM, AVERAGE};

/** @brief Reduce tensor to scalar.
 *
 *  @todo Reduction over specified dimensions.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class reduction_layer : public transform_layer {
private:

  /** Reduction mode. */
  const reduction_mode m_mode;

  /** Vector composed of ones. */
  DMat<Dev> m_ones;

public:

  reduction_layer(lbann_comm *comm,
                  reduction_mode mode)
    : transform_layer(comm),
      m_mode(mode) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "reduction currently only supports DATA_PARALLEL");
    if (mode == reduction_mode::INVALID) {
      LBANN_ERROR("invalid reduction mode");
    }
  }

  reduction_layer* copy() const override { return new reduction_layer(*this); }
  std::string get_type() const override { return "reduction"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = transform_layer::get_description();
    std::string mode_str;
    switch (m_mode) {
    case reduction_mode::SUM:     mode_str = "sum";     break;
    case reduction_mode::AVERAGE: mode_str = "average"; break;
    case reduction_mode::INVALID:
    default:
      mode_str = "invalid";
    }
    desc.add("Mode", mode_str);
    return desc;
  }

protected:

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims({1});
  }

  void fp_compute() override {

    // Local matrices
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();
    const El::Int input_size = local_input.Height();

    // Apply reduction
    switch (m_mode) {
    case reduction_mode::SUM:
      El::Ones(m_ones, input_size, 1);
      El::Gemv(El::TRANSPOSE,
               DataType(1), local_input, m_ones,
               DataType(0), local_output);
      break;
    case reduction_mode::AVERAGE:
      El::Ones(m_ones, input_size, 1);
      El::Gemv(El::TRANSPOSE,
               DataType(1) / input_size, local_input, m_ones,
               DataType(0), local_output);
      break;
    default:
      LBANN_ERROR("invalid reduction mode");
    }

  }

  void bp_compute() override {

    // Local matrices
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();
    const El::Int input_size = local_gradient_wrt_input.Height();

    // Compute gradients w.r.t. inputs
    switch (m_mode) {
    case reduction_mode::SUM:
      El::Ones(m_ones, input_size, 1);
      El::Gemm(El::NORMAL, El::NORMAL,
               DataType(1), m_ones, local_gradient_wrt_output,
               DataType(0), local_gradient_wrt_input);
      break;
    case reduction_mode::AVERAGE:
      El::Ones(m_ones, input_size, 1);
      El::Gemm(El::NORMAL, El::NORMAL,
               DataType(1) / input_size, m_ones, local_gradient_wrt_output,
               DataType(0), local_gradient_wrt_input);
      break;
    default:
      LBANN_ERROR("invalid reduction mode");
    }

  }

};

} // namespace lbann

#endif // LBANN_LAYER_REDUCTION_HPP_INCLUDED
