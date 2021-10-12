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

#define LBANN_INSTANCE_NORM_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/instance_norm.hpp"
#include "lbann/utils/exception.hpp"
#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

// =============================================
// Forward prop
// =============================================

namespace {

/** @brief Forward prop */
template <typename TensorDataType>
void fp_impl(lbann_comm& comm,
             El::Int num_channels,
             El::Int channel_size,
             TensorDataType epsilon,
             const El::AbstractDistMatrix<TensorDataType>& input,
             El::AbstractDistMatrix<TensorDataType>& output,
             El::Matrix<TensorDataType, El::Device::CPU>& local_workspace)
{
  // Local matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<LocalMat&>(output.Matrix());

  // Dimensions
  const El::Int local_mini_batch_size = local_input.Width();

  // Trivial case if channel size is 1
  // Note: Output is constant.
  if (channel_size <= 1) {
    El::Zero(output);
    return;
  }

  // Compute sums
<<<<<<< HEAD
  El::Zeros(local_workspace, 2 * num_channels, local_mini_batch_size);
  auto local_sums = El::View(local_workspace, El::IR(0, num_channels), El::ALL);
  auto local_sqsums =
    El::View(local_workspace, El::IR(num_channels, 2 * num_channels), El::ALL);
=======
  El::Zeros(local_workspace, 2*num_channels, local_mini_batch_size);
  auto local_sums = El::View(local_workspace,
                             El::IR(0, num_channels),
                             El::ALL);
  auto local_sqsums = El::View(local_workspace,
                               El::IR(num_channels, 2*num_channels),
                               El::ALL);


>>>>>>> 48d0878c1 (initial Caliper support/annotation)
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int k = 0; k < local_mini_batch_size; ++k) {
    for (El::Int j = 0; j < num_channels; ++j) {
      auto& sum = local_sums(j, k);
      auto& sqsum = local_sqsums(j, k);
      for (El::Int i = 0; i < channel_size; ++i) {
        const auto& x = local_input(i + j * channel_size, k);
        sum += x;
        sqsum += x * x;
      }
    }
  }

  // Normalize output
  //   mean = sum(x_i) / n
  //   var = ( sum(x_i^2)/n - mean^2 ) * n/(n-1)
  //   y_i = (x_i - mean) / sqrt(var + epsilon)
  const TensorDataType mean_scale = 1. / channel_size;
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int k = 0; k < local_mini_batch_size; ++k) {
    for (El::Int j = 0; j < num_channels; ++j) {
      const auto& sum = local_sums(j, k);
      const auto& sqsum = local_sqsums(j, k);
      const auto mean = sum * mean_scale;
      const auto sqmean = sqsum * mean_scale;
      auto var = (sqmean - mean * mean);
      var = std::max(var, TensorDataType{0.});
      const TensorDataType inv_stdev =
        TensorDataType{1.} / std::sqrt(var + epsilon);
      for (El::Int i = 0; i < channel_size; ++i) {
        const auto& x = local_input(i + j * channel_size, k);
        auto& y = local_output(i + j * channel_size, k);
        y = (x - mean) * inv_stdev;
      }
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void instance_norm_layer<TensorDataType, Layout, Device>::fp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("instance_norm_layer::fp_compute");
  const El::Int num_channels = this->get_output_dims().front();
  const El::Int channel_size = this->get_output_size() / num_channels;
  fp_impl(*this->get_comm(),
          num_channels,
          channel_size,
          this->m_epsilon,
          this->get_prev_activations(),
          this->get_activations(),
          this->m_workspace);
}

// =============================================
// Backprop
// =============================================

namespace {

/** @brief Backprop */
template <typename TensorDataType>
void bp_impl(lbann_comm& comm,
             El::Int num_channels,
             El::Int channel_size,
             TensorDataType epsilon,
             const El::AbstractDistMatrix<TensorDataType>& input,
             const El::AbstractDistMatrix<TensorDataType>& output_grad,
             El::AbstractDistMatrix<TensorDataType>& input_grad,
             const El::Matrix<TensorDataType, El::Device::CPU>& local_workspace)
{

  // Local matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  const auto& local_output_grad =
    dynamic_cast<const LocalMat&>(output_grad.LockedMatrix());
  auto& local_input_grad = dynamic_cast<LocalMat&>(input_grad.Matrix());
  const auto local_sums =
    El::LockedView(local_workspace, El::IR(0, num_channels), El::ALL);
  const auto local_sqsums =
    El::LockedView(local_workspace,
                   El::IR(num_channels, 2 * num_channels),
                   El::ALL);

  // Dimensions
  const El::Int local_mini_batch_size = local_input.Width();

  // Trivial case if channel size is 1
  // Note: Output is constant, so error signal is zero.
  if (channel_size <= 1) {
    El::Zero(input_grad);
    return;
  }

  // Compute gradient w.r.t. statistics
  //   dL/dmean = - sum(dL/dy_i) / sqrt(var+epsilon)
  //   dL/dvar = - sum(dL/dy_i * (x_i-mean)) * (var+epsilon)^(-3/2) / 2
  LocalMat local_statistics_grad;
  El::Zeros(local_statistics_grad, 2 * num_channels, local_mini_batch_size);
  auto local_means_grad =
    El::View(local_statistics_grad, El::IR(0, num_channels), El::ALL);
  auto local_vars_grad = El::View(local_statistics_grad,
                                  El::IR(num_channels, 2 * num_channels),
                                  El::ALL);
  const TensorDataType mean_scale = 1. / channel_size;

  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int k = 0; k < local_mini_batch_size; ++k) {
    for (El::Int j = 0; j < num_channels; ++j) {
      const auto& sum = local_sums(j, k);
      const auto& sqsum = local_sqsums(j, k);
      const auto mean = sum * mean_scale;
      const auto sqmean = sqsum * mean_scale;
      auto var = (sqmean - mean * mean);
      const TensorDataType inv_stdev =
        TensorDataType{1.} / std::sqrt(var + epsilon);
      auto& dmean = local_means_grad(j, k);
      auto& dvar = local_vars_grad(j, k);
      for (El::Int i = 0; i < channel_size; ++i) {
        const auto& x = local_input(i + j * channel_size, k);
        const auto& dy = local_output_grad(i + j * channel_size, k);
        dmean += dy;
        dvar += dy * (x - mean);
      }
      dmean *= -inv_stdev;
      dvar *= -inv_stdev * inv_stdev * inv_stdev / 2;
    }
  }

  // Compute gradient w.r.t. input
  //   dL/dx_i = ( dL/dy_i / sqrt(var+epsilon)
  //             + dL/dmean / n
  //             + dL/dvar * (x_i - mean) * 2/(n-1) )
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int k = 0; k < local_mini_batch_size; ++k) {
    for (El::Int j = 0; j < num_channels; ++j) {
      const auto& sum = local_sums(j, k);
      const auto& sqsum = local_sqsums(j, k);
      const auto mean = sum * mean_scale;
      const auto sqmean = sqsum * mean_scale;
      auto var = (sqmean - mean * mean);
      const TensorDataType inv_stdev =
        TensorDataType{1.} / std::sqrt(var + epsilon);
      const auto& dmean = local_means_grad(j, k);
      const auto& dvar = local_vars_grad(j, k);
      for (El::Int i = 0; i < channel_size; ++i) {
        const auto& x = local_input(i + j * channel_size, k);
        const auto& dy = local_output_grad(i + j * channel_size, k);
        auto& dx = local_input_grad(i + j * channel_size, k);
        dx = (dy * inv_stdev + dmean / channel_size +
              dvar * (x - mean) * 2 / channel_size);
      }
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void instance_norm_layer<TensorDataType, Layout, Device>::bp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("instance_norm_layer::bp_compute");
  const El::Int num_channels = this->get_output_dims().front();
  const El::Int channel_size = this->get_output_size() / num_channels;
  bp_impl(*this->get_comm(),
          num_channels,
          channel_size,
          this->m_epsilon,
          this->get_prev_activations(),
          this->get_prev_error_signals(),
          this->get_error_signals(),
          this->m_workspace);
}

// =============================================
// Builder function
// =============================================

namespace {

template <typename T, data_layout L, El::Device D>
struct Builder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR("Attempted to construct instance_norm_layer ",
                "with invalid parameters ",
                "(TensorDataType=",
                TypeName<T>(),
                ", ",
                "Layout=",
                to_string(L),
                ", ",
                "Device=",
                to_string(D),
                ")");
    return nullptr;
  }
};

template <El::Device Device>
struct Builder<float, data_layout::DATA_PARALLEL, Device>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType =
      instance_norm_layer<float, data_layout::DATA_PARALLEL, Device>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
  }
};

template <El::Device Device>
struct Builder<double, data_layout::DATA_PARALLEL, Device>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType =
      instance_norm_layer<double, data_layout::DATA_PARALLEL, Device>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
  }
};

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_instance_norm_layer_from_pbuf(lbann_comm* comm,
                                    lbann_data::Layer const& proto_layer)
{
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  const auto& params = proto_layer.instance_norm();
  const double epsilon = params.has_epsilon() ? params.epsilon().value() : 1e-5;
  return BuilderType::Build(El::To<TensorDataType>(epsilon));
}

// =============================================
// Explicit template instantiation
// =============================================

#define PROTO(T)                                                               \
  template class instance_norm_layer<T,                                        \
                                     data_layout::DATA_PARALLEL,               \
                                     El::Device::CPU>
#include "lbann/macros/instantiate.hpp"
#undef PROTO

#ifdef LBANN_HAS_GPU
#define PROTO(T)                                                               \
  extern template class instance_norm_layer<T,                                 \
                                            data_layout::DATA_PARALLEL,        \
                                            El::Device::GPU>
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_HAS_GPU

#define PROTO_DEVICE(T, Device)                                                \
  LBANN_LAYER_BUILDER_ETI(instance_norm, T, Device)
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

} // namespace lbann
