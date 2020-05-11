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
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class reduction_layer : public transform_layer<TensorDataType> {
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "reduction currently only supports DATA_PARALLEL");
private:

  /** Reduction mode. */
  const reduction_mode m_mode;

  /** Vector composed of ones. */
  El::Matrix<TensorDataType, Dev> m_ones;

public:

  reduction_layer(lbann_comm *comm,
                  reduction_mode mode)
    : transform_layer<TensorDataType>(comm),
      m_mode(mode) {
    if (mode == reduction_mode::INVALID) {
      LBANN_ERROR("invalid reduction mode");
    }
  }

  reduction_layer* copy() const override { return new reduction_layer(*this); }
  std::string get_type() const override { return "reduction"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = transform_layer<TensorDataType>::get_description();
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

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims({1});
  }

  void fp_compute() override {

    // Local matrices
    const auto& local_input = this->get_local_prev_activations();
    auto& local_output = this->get_local_activations();
    const El::Int input_size = local_input.Height();

    // Apply reduction
    switch (m_mode) {
    case reduction_mode::SUM:
      El::Ones(m_ones, input_size, 1);
      El::Gemv(El::TRANSPOSE,
               El::TypeTraits<TensorDataType>::One(), local_input, m_ones,
               El::TypeTraits<TensorDataType>::Zero(), local_output);
      break;
    case reduction_mode::AVERAGE:
      El::Ones(m_ones, input_size, 1);
      El::Gemv(El::TRANSPOSE,
               El::TypeTraits<TensorDataType>::One() / El::To<TensorDataType>(input_size),
               local_input, m_ones,
               El::TypeTraits<TensorDataType>::Zero(), local_output);
      break;
    default:
      LBANN_ERROR("invalid reduction mode");
    }

  }

  void bp_compute() override {

    // Local matrices
    const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
    auto& local_gradient_wrt_input = this->get_local_error_signals();
    const El::Int input_size = local_gradient_wrt_input.Height();

    // Compute gradients w.r.t. inputs
    switch (m_mode) {
    case reduction_mode::SUM:
      El::Ones(m_ones, input_size, 1);
      El::Gemm(El::NORMAL, El::NORMAL,
               El::TypeTraits<TensorDataType>::One(), m_ones, local_gradient_wrt_output,
               El::TypeTraits<TensorDataType>::Zero(), local_gradient_wrt_input);
      break;
    case reduction_mode::AVERAGE:
      El::Ones(m_ones, input_size, 1);
      El::Gemm(El::NORMAL, El::NORMAL,
               El::TypeTraits<TensorDataType>::One() / El::To<TensorDataType>(input_size),
               m_ones, local_gradient_wrt_output,
               El::TypeTraits<TensorDataType>::Zero(), local_gradient_wrt_input);
      break;
    default:
      LBANN_ERROR("invalid reduction mode");
    }

  }

};

#ifndef LBANN_REDUCTION_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class reduction_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_REDUCTION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_REDUCTION_HPP_INCLUDED
