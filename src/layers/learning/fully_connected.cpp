////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#define LBANN_FULLY_CONNECTED_LAYER_INSTANTIATE
#include "lbann/layers/learning/fully_connected.hpp"

#include "lbann/optimizers/optimizer.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"

#include "lbann/proto/datatype_helpers.hpp"

#include "lbann/proto/layers.pb.h"

#include <sstream>
#include <string>

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::fully_connected_layer(
  int output_size,
  bool transpose,
  WeightsType* weight,
  bool has_bias)
  : data_type_layer<TensorDataType>(nullptr),
    m_bias_gradient(nullptr),
    m_transpose(transpose)
{

  // Initialize output tensor dimensions
  this->set_output_dims({output_size});

  // Initialize bias
  m_bias_scaling_factor = (has_bias ? El::TypeTraits<TensorDataType>::One()
                                    : El::TypeTraits<TensorDataType>::Zero());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::fully_connected_layer()
  : fully_connected_layer(0, false, nullptr, false)
{}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::fully_connected_layer(
  const fully_connected_layer& other)
  : data_type_layer<TensorDataType>(other),
    m_bias_scaling_factor(other.m_bias_scaling_factor),
    m_transpose(other.m_transpose)
{

  // Deep matrix copies
  m_bias_gradient = other.m_bias_gradient;
  if (m_bias_gradient != nullptr) {
    m_bias_gradient = m_bias_gradient->Copy();
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
auto fully_connected_layer<TensorDataType, T_layout, Dev>::operator=(
  const fully_connected_layer& other) -> fully_connected_layer&
{
  data_type_layer<TensorDataType>::operator=(other);
  m_bias_scaling_factor = other.m_bias_scaling_factor;
  m_transpose = other.m_transpose;

  // Deep matrix copies
  deallocate_matrices();
  m_bias_gradient = other.m_bias_gradient;
  if (m_bias_gradient != nullptr) {
    m_bias_gradient = m_bias_gradient->Copy();
  }

  return *this;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::~fully_connected_layer()
{
  deallocate_matrices();
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
description
fully_connected_layer<TensorDataType, T_layout, Dev>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  const auto& bias_str =
    (m_bias_scaling_factor == El::TypeTraits<TensorDataType>::Zero()
       ? "disabled"
       : "enabled");
  desc.add("Bias", bias_str);
  return desc;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Initialize default weights if none are provided
  if (this->num_weights() > 2) {
    LBANN_ERROR("attempted to setup ",
                this->get_name(),
                " with an invalid number of weights");
  }
  if (m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    this->set_num_weights(2);
  }
  else {
    this->set_num_weights(1);
  }
  if (!this->has_weights(0)) {
    auto w = std::make_shared<WeightsType>(*this->get_comm());
    auto init = std::make_unique<he_initializer<TensorDataType>>(
      probability_distribution::gaussian);
    auto opt = this->m_model->template create_optimizer<TensorDataType>();
    w->set_name(this->get_name() + "_linearity_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->set_weights(0, w);
    this->m_model->add_weights(std::move(w));
  }
  auto& linearity_weights = this->get_weights(0);

  // Initialize variance scaling initialization
  if (auto* initializer = linearity_weights.get_initializer()) {
    set_fan_in(*initializer, this->get_input_size());
    set_fan_out(*initializer, this->get_output_size());
  }

  // Input and output dimensions
  const auto& input_dims_ = this->get_input_dims();
  const auto& output_dims_ = this->get_output_dims();
  std::vector<size_t> input_dims(input_dims_.begin(), input_dims_.end());
  std::vector<size_t> output_dims(output_dims_.begin(), output_dims_.end());

  // Setup linearity weights
  auto linearity_dist = this->get_prev_activations().DistData();
  if (linearity_dist.colDist != El::MC || linearity_dist.rowDist != El::MR) {
    linearity_dist.colDist = El::STAR;
    linearity_dist.rowDist = El::STAR;
  }
  if (m_transpose) {
    linearity_weights.set_dims(input_dims, output_dims);
  }
  else {
    linearity_weights.set_dims(output_dims, input_dims);
  }
  linearity_weights.set_matrix_distribution(linearity_dist);

  // Set up bias if needed.
  if (m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    if (!this->has_weights(1)) {
      auto w = std::make_shared<WeightsType>(*this->get_comm());
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_bias_weights");
      w->set_optimizer(std::move(opt));
      this->set_weights(1, w);
      this->m_model->add_weights(std::move(w));
    }
    auto& bias_weights = this->get_weights(1);
    // Setup bias weights
    auto bias_dist = this->get_activations().DistData();
    bias_dist.rowDist = El::STAR;
    bias_weights.set_dims(output_dims);
    bias_weights.set_matrix_distribution(bias_dist);

    // Setup bias gradient
    if (Dev == El::Device::CPU) {
      if (T_layout == data_layout::MODEL_PARALLEL) {
        // Allocate a MCStarMat (RowSumMat)
        this->m_bias_gradient =
          new El::DistMatrix<TensorDataType,
                             El::MC,
                             El::STAR,
                             El::ELEMENT,
                             El::Device::CPU>(*bias_dist.grid);
      }
      else if (T_layout == data_layout::DATA_PARALLEL) {
        // Allocate a StarMat
        this->m_bias_gradient =
          new El::DistMatrix<TensorDataType,
                             El::STAR,
                             El::STAR,
                             El::ELEMENT,
                             El::Device::CPU>(*bias_dist.grid);
      }
    }
    if (this->m_bias_gradient != nullptr) {
      El::Zeros(*this->m_bias_gradient,
                bias_weights.get_matrix_height(),
                bias_weights.get_matrix_width());
    }
  }

  // Initialize freeze state
  auto const num_weights = this->num_weights();
  for (size_t ii = 0; ii < num_weights; ++ii) {
    auto& w = this->get_weights(ii);
    if (this->m_frozen) {
      w.freeze();
    }
    else {
      w.unfreeze();
    }
  }
  for (size_t ii = 0; ii < num_weights; ++ii) {
    auto& w = this->get_weights(ii);
    if (w.is_frozen() != this->is_frozen()) {
      LBANN_ERROR((this->is_frozen() ? "" : "un"),
                  "frozen ",
                  "layer \"",
                  this->get_name(),
                  "\" has ",
                  (w.is_frozen() ? "" : "un"),
                  "frozen ",
                  "weights \"",
                  w.get_name(),
                  "\"");
    }
  }
}

/** CPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::MODEL_PARALLEL,
                                           El::Device::CPU>& l)
{

  // Matrices
  const auto& input = l.get_prev_activations();
  auto& output = l.get_activations();

  // Apply linearity
  // Note: Perform GEMMs independently if possible
  const auto& linearity = l.weights_values(0);
  if (!linearity.Participating()) {
    return;
  }
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity.LockedMatrix(),
             input.LockedMatrix(),
             El::TypeTraits<TensorDataType>::Zero(),
             output.Matrix());
  }
  else {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity,
             input,
             El::TypeTraits<TensorDataType>::Zero(),
             output);
  }

  // Apply bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& local_bias = l.weights_values(1).LockedMatrix();
    auto& local_output = output.Matrix();
    El::IndexDependentMap(
      local_output,
      (std::function<TensorDataType(
         El::Int,
         El::Int,
         const TensorDataType&)>)([&l, &local_bias](
                                    El::Int r,
                                    El::Int c,
                                    const TensorDataType& z) -> TensorDataType {
        return z + l.m_bias_scaling_factor * local_bias(r, 0);
      }));
  }
}

/** CPU implementation of backward prop computation. */
template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::MODEL_PARALLEL,
                                           El::Device::CPU>& l)
{

  // Matrices
  const auto& linearity = l.weights_values(0);
  const auto& input = l.get_prev_activations();
  const auto& gradient_wrt_output = l.get_prev_error_signals();
  auto& gradient_wrt_input = l.get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();
  if (!linearity.Participating()) {
    return;
  }

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_weights(1).get_optimizer();
    if (bias_optimizer != nullptr) {
      El::RowSum(local_gradient_wrt_output, l.m_bias_gradient->Matrix());
      bias_optimizer->add_to_gradient(*l.m_bias_gradient,
                                      l.m_bias_scaling_factor,
                                      true);
    }
  }

  // Compute gradient w.r.t. linearity if needed
  // Note: Perform GEMMs independently if possible
  auto* linearity_optimizer = l.get_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                   gradient_scale = El::TypeTraits<TensorDataType>::One();
    if (linearity.DistSize() == 1) {
      auto& linearity_gradient =
        linearity_optimizer->get_gradient_buffer(dst_scale,
                                                 gradient_scale,
                                                 true);
      if (l.m_transpose) {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 local_input,
                 local_gradient_wrt_output,
                 dst_scale,
                 linearity_gradient.Matrix());
      }
      else {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 local_gradient_wrt_output,
                 local_input,
                 dst_scale,
                 linearity_gradient.Matrix());
      }
    }
    else {
      auto& linearity_gradient =
        linearity_optimizer->get_gradient_buffer(dst_scale, gradient_scale);
      if (l.m_transpose) {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 input,
                 gradient_wrt_output,
                 dst_scale,
                 linearity_gradient);
      }
      else {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 gradient_wrt_output,
                 input,
                 dst_scale,
                 linearity_gradient);
      }
    }
  }

  // Compute gradient w.r.t. input
  // Note: Perform GEMMs independently if possible
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             local_linearity,
             local_gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(),
             local_gradient_wrt_input);
  }
  else {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity,
             gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(),
             gradient_wrt_input);
  }
}

/** CPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::DATA_PARALLEL,
                                           El::Device::CPU>& l)
{

  // Matrices
  const auto& local_input = l.get_local_prev_activations();
  auto& local_output = l.get_local_activations();

  // Apply linearity
  const auto& local_linearity = l.weights_values(0).LockedMatrix();
  El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           El::TypeTraits<TensorDataType>::One(),
           local_linearity,
           local_input,
           El::TypeTraits<TensorDataType>::Zero(),
           local_output);

  // Apply bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& local_bias = l.weights_values(1).LockedMatrix();
    El::IndexDependentMap(
      local_output,
      (std::function<TensorDataType(
         El::Int,
         El::Int,
         const TensorDataType&)>)([&l, &local_bias](
                                    El::Int r,
                                    El::Int c,
                                    const TensorDataType& z) -> TensorDataType {
        return z + l.m_bias_scaling_factor * local_bias(r, 0);
      }));
  }
}

/** CPU implementation of backward prop computation. */
template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::DATA_PARALLEL,
                                           El::Device::CPU>& l)
{

  // Matrices
  const auto& local_linearity = l.weights_values(0).LockedMatrix();
  const auto& local_input = l.get_local_prev_activations();
  const auto& local_gradient_wrt_output = l.get_local_prev_error_signals();
  auto& local_gradient_wrt_input = l.get_local_error_signals();

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_weights(1).get_optimizer();
    if (bias_optimizer != nullptr) {
      El::RowSum(local_gradient_wrt_output, l.m_bias_gradient->Matrix());
      bias_optimizer->add_to_gradient(*l.m_bias_gradient,
                                      l.m_bias_scaling_factor,
                                      true);
    }
  }

  // Compute gradient w.r.t. linearity if needed
  auto* linearity_optimizer = l.get_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                   gradient_scale = El::TypeTraits<TensorDataType>::Zero();
    auto& linearity_gradient =
      linearity_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);
    if (l.m_transpose) {
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               gradient_scale,
               local_input,
               local_gradient_wrt_output,
               dst_scale,
               linearity_gradient.Matrix());
    }
    else {
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               gradient_scale,
               local_gradient_wrt_output,
               local_input,
               dst_scale,
               linearity_gradient.Matrix());
    }
  }

  // Compute gradient w.r.t. input
  El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
           El::NORMAL,
           El::TypeTraits<TensorDataType>::One(),
           local_linearity,
           local_gradient_wrt_output,
           El::TypeTraits<TensorDataType>::Zero(),
           local_gradient_wrt_input);
}

#ifdef LBANN_HAS_GPU
/** GPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::DATA_PARALLEL,
                                           El::Device::GPU>& l)
{

  // Matrices
  const auto& local_input = l.get_local_prev_activations();
  auto& local_output = l.get_local_activations();

  // Apply linearity
  const auto& local_linearity = l.weights_values(0).LockedMatrix();
  El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           El::TypeTraits<TensorDataType>::One(),
           local_linearity,
           local_input,
           El::TypeTraits<TensorDataType>::Zero(),
           local_output);

  // Apply bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& local_bias = l.weights_values(1).LockedMatrix();
    El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
    ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif                     // HYDROGEN_HAVE_CUB
    ones.Resize(local_input.Width(), 1);
    El::Fill(ones, El::TypeTraits<TensorDataType>::One());
    El::Gemm(El::NORMAL,
             El::TRANSPOSE,
             l.m_bias_scaling_factor,
             local_bias,
             ones,
             El::TypeTraits<TensorDataType>::One(),
             local_output);
  }
}

/** GPU implementation of backward prop computation. */
template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::DATA_PARALLEL,
                                           El::Device::GPU>& l)
{

  // Matrices
  const auto& local_linearity = l.weights_values(0).LockedMatrix();
  const auto& local_input = l.get_local_prev_activations();
  const auto& local_gradient_wrt_output = l.get_local_prev_error_signals();
  auto& local_gradient_wrt_input = l.get_local_error_signals();

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_weights(1).get_optimizer();
    if (bias_optimizer != nullptr) {
      TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                     gradient_scale = El::TypeTraits<TensorDataType>::Zero();
      auto& bias_gradient =
        bias_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);
      if (local_gradient_wrt_output.Height() < 1 ||
          local_gradient_wrt_output.Width() < 1) {
        El::Scale(dst_scale, bias_gradient);
      }
      else {
        El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
        ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif                         // HYDROGEN_HAVE_CUB
        ones.Resize(local_gradient_wrt_output.Width(), 1);
        El::Fill(ones, El::TypeTraits<TensorDataType>::One());
        El::Gemv(El::NORMAL,
                 gradient_scale,
                 local_gradient_wrt_output,
                 ones,
                 dst_scale,
                 bias_gradient.Matrix());
      }
    }
  }

  // Compute gradient w.r.t. linearity if needed
  auto* linearity_optimizer = l.get_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                   gradient_scale = El::TypeTraits<TensorDataType>::Zero();
    auto& linearity_gradient =
      linearity_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);
    if (l.m_transpose) {
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               gradient_scale,
               local_input,
               local_gradient_wrt_output,
               dst_scale,
               linearity_gradient.Matrix());
    }
    else {
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               gradient_scale,
               local_gradient_wrt_output,
               local_input,
               dst_scale,
               linearity_gradient.Matrix());
    }
  }

  // Compute gradient w.r.t. input
  El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
           El::NORMAL,
           El::TypeTraits<TensorDataType>::One(),
           local_linearity,
           local_gradient_wrt_output,
           El::TypeTraits<TensorDataType>::Zero(),
           local_gradient_wrt_input);
}

template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::MODEL_PARALLEL,
                                           El::Device::GPU>& l)
{

  // Matrices
  const auto& input = l.get_prev_activations();
  auto& output = l.get_activations();

  // Apply linearity
  // Note: Perform GEMMs independently if possible
  const auto& linearity = l.weights_values(0);
  if (!linearity.Participating()) {
    return;
  }
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity.LockedMatrix(),
             input.LockedMatrix(),
             El::TypeTraits<TensorDataType>::Zero(),
             output.Matrix());
  }
  else {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity,
             input,
             El::TypeTraits<TensorDataType>::Zero(),
             output);
  }

  // Apply bias if needed
  // Note: local outer product is sufficient, no need for global GEMM
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& bias = l.weights_values(1);
    El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
    ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif                     // HYDROGEN_HAVE_CUB
    ones.Resize(input.LocalWidth(), 1);
    El::Fill(ones, El::TypeTraits<TensorDataType>::One());
    El::Gemm(El::NORMAL,
             El::TRANSPOSE,
             l.m_bias_scaling_factor,
             bias.LockedMatrix(),
             ones,
             El::TypeTraits<TensorDataType>::One(),
             output.Matrix());
  }
}

template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::MODEL_PARALLEL,
                                           El::Device::GPU>& l)
{

  // Matrices
  const auto& linearity = l.weights_values(0);
  const auto& input = l.get_prev_activations();
  const auto& gradient_wrt_output = l.get_prev_error_signals();
  auto& gradient_wrt_input = l.get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();
  if (!linearity.Participating()) {
    return;
  }

  // Compute gradient w.r.t. bias if needed
  // Note: local GEMV is sufficient, no need for global row sum
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_weights(1).get_optimizer();
    if (bias_optimizer != nullptr) {
      TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                     gradient_scale = El::TypeTraits<TensorDataType>::Zero();
      auto& bias_gradient =
        bias_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);
      if (local_gradient_wrt_output.Height() < 1 ||
          local_gradient_wrt_output.Width() < 1) {
        El::Scale(dst_scale, bias_gradient);
      }
      else {
        El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
        ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif                         // HYDROGEN_HAVE_CUB
        ones.Resize(local_gradient_wrt_output.Width(), 1);
        El::Fill(ones, El::TypeTraits<TensorDataType>::One());
        El::Gemv(El::NORMAL,
                 gradient_scale,
                 local_gradient_wrt_output,
                 ones,
                 dst_scale,
                 bias_gradient.Matrix());
      }
    }
  }

  // Compute gradient w.r.t. linearity if needed
  // Note: Perform GEMMs independently if possible
  auto* linearity_optimizer = l.get_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                   gradient_scale = El::TypeTraits<TensorDataType>::Zero();
    if (linearity.DistSize() == 1) {
      auto& linearity_gradient =
        linearity_optimizer->get_gradient_buffer(dst_scale,
                                                 gradient_scale,
                                                 true);
      if (l.m_transpose) {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 local_input,
                 local_gradient_wrt_output,
                 dst_scale,
                 linearity_gradient.Matrix());
      }
      else {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 local_gradient_wrt_output,
                 local_input,
                 dst_scale,
                 linearity_gradient.Matrix());
      }
    }
    else {
      auto& linearity_gradient =
        linearity_optimizer->get_gradient_buffer(dst_scale, gradient_scale);
      if (l.m_transpose) {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 input,
                 gradient_wrt_output,
                 dst_scale,
                 linearity_gradient);
      }
      else {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 gradient_wrt_output,
                 input,
                 dst_scale,
                 linearity_gradient);
      }
    }
  }

  // Compute gradient w.r.t. input
  // Note: Perform GEMMs independently if possible
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             local_linearity,
             local_gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(),
             local_gradient_wrt_input);
  }
  else {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity,
             gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(),
             gradient_wrt_input);
  }
}

#endif // LBANN_HAS_GPU

template <typename T, data_layout L, El::Device D>
void fully_connected_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_fully_connected();
  msg->set_num_neurons(get_linear_size(this->get_output_dims()));
  auto const has_bias = (this->num_weights() > 1UL);
  msg->set_has_bias(has_bias);
  msg->set_transpose(m_transpose);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>::fp_compute()
{
  fp_compute_impl<TensorDataType>(*this);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>::bp_compute()
{
  bp_compute_impl<TensorDataType>(*this);
}

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void fully_connected_layer<T, L, D>::fill_onnx_node(
  onnx::GraphProto& graph) const
{
  auto const& parent = this->get_parent_layer(0);
  auto const has_bias = (this->num_weights() > 1UL);

  // Setup the inputs.
  auto const layer_name = this->get_name();
  auto const A_name = layer_name + "_" + parent.get_name() + "_reshape";
  auto const B_name = this->get_weights(0).get_name();
  auto const C_name = (has_bias ? layer_name + "_bias_reshape" : std::string{});

  // Flatten the input tensor (A, wrt GEMM).
  {
    auto* flatten_input_shape = graph.add_initializer();
    flatten_input_shape->set_name(layer_name + "_" + parent.get_name() +
                                  "_shape");
    flatten_input_shape->set_data_type(onnx::TensorProto::INT64);
    flatten_input_shape->add_dims(2);
    flatten_input_shape->add_int64_data(0);
    flatten_input_shape->add_int64_data(-1);
    flatten_input_shape->set_doc_string(
      "Shape for " + layer_name + " reshape " + parent.get_name() + " node");

    onnx::NodeProto* flatten_input = graph.add_node();
    flatten_input->add_input(
      parent.get_name() + "_" +
      std::to_string(parent.find_child_layer_index(*this)));
    flatten_input->add_input(flatten_input_shape->name());
    flatten_input->add_output(A_name);
    flatten_input->set_name(A_name);
    flatten_input->set_op_type("Reshape");
    flatten_input->set_domain("");
    flatten_input->set_doc_string("Reshape " + parent.get_name() +
                                  " for Fully Connected Layer");
  }

  // Setup the bias node, if applicable.
  if (has_bias) {
    // bias = Reshape(data=bias, shape=[1,-1])
    auto* shape = graph.add_initializer();
    shape->set_name(this->get_name() + "_bias_shape");
    shape->set_data_type(onnx::TensorProto::INT64);
    shape->add_dims(2);
    shape->add_int64_data(1);
    shape->add_int64_data(-1);
    shape->set_doc_string("Shape for " + layer_name + " Bias");

    auto* bias = graph.add_node();
    bias->add_input(this->get_weights(1).get_name());
    bias->add_input(shape->name());
    bias->add_output(this->get_name() + "_bias_reshape");
    bias->set_name(this->get_name() + "_bias_reshape");
    bias->set_op_type("Reshape");
    bias->set_domain("");
    bias->set_doc_string("Reshape bias for Fully Connected Layer");
  }

  auto* gemm = graph.add_node();
  gemm->add_input(A_name);
  gemm->add_input(B_name);
  if (has_bias)
    gemm->add_input(C_name);

  gemm->add_output(layer_name + "_0");
  gemm->set_name(layer_name + "_0");
  gemm->set_op_type("Gemm");
  gemm->set_domain("");
  gemm->set_doc_string("Gemm node for Fully Connected Layer");

  {
    auto* alpha = gemm->add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(onnx::AttributeProto::FLOAT);
    alpha->set_f(1);
  }
  {
    auto* beta = gemm->add_attribute();
    beta->set_name("beta");
    beta->set_type(onnx::AttributeProto::FLOAT);
    beta->set_f(has_bias ? 1 : 0);
  }
  {
    auto* transA = gemm->add_attribute();
    transA->set_name("transA");
    transA->set_type(onnx::AttributeProto::INT);
    transA->set_i(m_transpose ? 1 : 0);
  }
  {
    auto* transB = gemm->add_attribute();
    transB->set_name("transB");
    transB->set_type(onnx::AttributeProto::INT);
    transB->set_i(1); // Should be 1 because ONNX will do x*W^T + b
  }
}
#endif // LBANN_HAS_ONNX

template <typename TensorDataType, data_layout layout, El::Device device>
std::unique_ptr<Layer>
build_fully_connected_layer_from_pbuf(lbann_comm* comm,
                                      lbann_data::Layer const& layer_msg)
{
  using LayerType = fully_connected_layer<TensorDataType, layout, device>;
  const auto& params = layer_msg.fully_connected();
  return std::make_unique<LayerType>(params.num_neurons(),
                                     params.transpose(),
                                     nullptr,
                                     params.has_bias());
}

#define PROTO_DEVICE(T, Device)                                                \
  template class fully_connected_layer<T, data_layout::DATA_PARALLEL, Device>; \
  template class fully_connected_layer<T,                                      \
                                       data_layout::MODEL_PARALLEL,            \
                                       Device>;                                \
  template std::unique_ptr<Layer>                                              \
  build_fully_connected_layer_from_pbuf<T,                                     \
                                        data_layout::DATA_PARALLEL,            \
                                        Device>(lbann_comm*,                   \
                                                lbann_data::Layer const&);     \
  template std::unique_ptr<Layer>                                              \
  build_fully_connected_layer_from_pbuf<T,                                     \
                                        data_layout::MODEL_PARALLEL,           \
                                        Device>(lbann_comm*,                   \
                                                lbann_data::Layer const&)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
