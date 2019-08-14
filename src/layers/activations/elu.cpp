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

#define ELU_LAYER_INSTANTIATE
#include "lbann/layers/activations/elu.hpp"

namespace lbann {

namespace {

// Useful constants
constexpr DataType zero = 0;

/** Local forward prop computation. */
void local_fp(DataType alpha,
              const AbsMat& input,
              AbsMat& output) {
  const auto& height = input.Height();
  const auto& width = input.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = input(row, col);
      auto& y = output(row, col);
      y = (x > zero) ? x : alpha * std::expm1(x);
    }
  }
}

/** Local backprop computation. */
void local_bp(DataType alpha,
              const AbsMat& input,
              const AbsMat& gradient_wrt_output,
              AbsMat& gradient_wrt_input) {
  const auto& height = input.Height();
  const auto& width = input.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = input(row, col);
      const auto& dy = gradient_wrt_output(row, col);
      auto& dx = gradient_wrt_input(row, col);
      dx = (x > zero) ? dy : dy * alpha * std::exp(x);
    }
  }
}

} // namespace

// COMMON IMPL
template <data_layout Layout, El::Device Device>
elu_layer<Layout, Device>::elu_layer(lbann_comm *comm, DataType alpha)
  : Layer(comm) , m_alpha(alpha)
{}

template <data_layout Layout, El::Device Device>
auto elu_layer<Layout, Device>::copy() const -> elu_layer* {
  return new elu_layer(*this);
}

template <data_layout Layout, El::Device Device>
std::string elu_layer<Layout, Device>::get_type() const {
  return "ELU";
}

template <data_layout Layout, El::Device Device>
data_layout elu_layer<Layout, Device>::get_data_layout() const {
  return Layout;
}

template <data_layout Layout, El::Device Device>
El::Device elu_layer<Layout, Device>::get_device_allocation() const {
  return Device;
}

template <data_layout Layout, El::Device Device>
description elu_layer<Layout, Device>::get_description() const {
  auto desc = Layer::get_description();
  desc.add("alpha", m_alpha);
  return desc;
}

template <data_layout Layout, El::Device Device>
void elu_layer<Layout, Device>::setup_dims() {
  Layer::setup_dims();
  set_output_dims(get_input_dims());
}

// SPECIAL IMPL

template <>
void elu_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
       ::fp_compute() {
  local_fp(m_alpha,
           get_local_prev_activations(),
           get_local_activations());
}
template <>
void elu_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::bp_compute() {
  local_bp(m_alpha,
           get_local_prev_activations(),
           get_local_prev_error_signals(),
           get_local_error_signals());
}
template <>
void elu_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
       ::fp_compute() {
  local_fp(m_alpha,
           get_local_prev_activations(),
           get_local_activations());
}
template <>
void elu_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
     ::bp_compute() {
  local_bp(m_alpha,
           get_local_prev_activations(),
           get_local_prev_error_signals(),
           get_local_error_signals());
}

} // namespace lbann

template class lbann::elu_layer<lbann::data_layout::DATA_PARALLEL, El::Device::CPU>;
template class lbann::elu_layer<lbann::data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
template class lbann::elu_layer<lbann::data_layout::DATA_PARALLEL, El::Device::GPU>;
template class lbann::elu_layer<lbann::data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
