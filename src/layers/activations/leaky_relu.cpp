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

#include "lbann/layers/activations/leaky_relu.hpp"

namespace lbann {

namespace {

// Useful constants
constexpr DataType zero = 0;

/** Local forward prop computation. */
void local_fp(DataType negative_slope,
              const AbsMat& input,
              AbsMat& output) {
  const auto& height = input.Height();
  const auto& width = input.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = input(row, col);
      auto& y = output(row, col);
      y = (x > zero) ? x : negative_slope * x;
    }
  }
}

/** Local backprop computation. */
void local_bp(DataType negative_slope,
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
      dx = (x > zero) ? dy : negative_slope * dy;
    }
  }
}

} // namespace

template <>
void leaky_relu_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
       ::fp_compute() {
  local_fp(m_negative_slope,
           get_local_prev_activations(),
           get_local_activations());
}
template <>
void leaky_relu_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::bp_compute() {
  local_bp(m_negative_slope,
           get_local_prev_activations(),
           get_local_prev_error_signals(),
           get_local_error_signals());
}
template <>
void leaky_relu_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
       ::fp_compute() {
  local_fp(m_negative_slope,
           get_local_prev_activations(),
           get_local_activations());
}
template <>
void leaky_relu_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
     ::bp_compute() {
  local_bp(m_negative_slope,
           get_local_prev_activations(),
           get_local_prev_error_signals(),
           get_local_error_signals());
}

#ifdef LBANN_HAS_DISTCONV
using namespace dc;

template <>
void leaky_relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::setup_tensor_distribution_init(
    std::map<const Layer*, std::array<Dist, dc::num_dists>> &dists,
    std::map<Dist*, std::set<Dist*>> &invariants,
    std::set<Dist*> &updated,
    std::set<Dist*> &fixed)  {
  Layer::setup_tensor_distribution_init(dists, invariants, updated, fixed);
  if (!distconv_enabled()) return;
  auto &layer_dists = dists[this];
  // x == y
  invariants[&layer_dists[0]].insert(&layer_dists[1]);
  invariants[&layer_dists[1]].insert(&layer_dists[0]);
  // x == dx
  invariants[&layer_dists[0]].insert(&layer_dists[2]);
  invariants[&layer_dists[2]].insert(&layer_dists[0]);
  // dx == dy
  invariants[&layer_dists[2]].insert(&layer_dists[3]);
  invariants[&layer_dists[3]].insert(&layer_dists[2]);
}

template <>
void leaky_relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::
setup_tensors_fwd(const std::array<Dist, dc::num_dists> &dists) {
  Layer::setup_tensors_fwd(dists);
  if (!distconv_enabled()) return;
  setup_prev_activations_tensor(dists);
  setup_activations_tensor(dists);
  setup_activations_copyout_tensor(dists);
}

template <>
void leaky_relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::
setup_tensors_bwd(const std::array<Dist, dc::num_dists> &dists)  {
  Layer::setup_tensors_bwd(dists);
  if (!distconv_enabled()) return;
  setup_prev_error_signals_tensor(dists);
  setup_error_signals_tensor(dists);
  setup_error_signals_copyout_tensor(dists);
  m_leaky_relu = new LeakyReLU(get_backend());
}

template <>
void leaky_relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::
fp_compute_distconv() {
  MPIPrintStreamDebug() << get_name() << ": " << __FUNCTION__;
  assert_always(distconv_enabled());
  m_leaky_relu->forward(m_prev_activations_t, m_negative_slope, m_activations_t);
  copy_out_activations();
}

template <>
void leaky_relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::
bp_compute_distconv() {
  MPIPrintStreamDebug() << get_name() << ": " << __FUNCTION__;
  assert_always(distconv_enabled());
  m_leaky_relu->backward(m_prev_activations_t, m_prev_error_signals_t,
                         m_negative_slope, m_error_signals_t);
  copy_out_error_signals();
}

#endif // LBANN_HAS_DISTCONV

} // namespace lbann
