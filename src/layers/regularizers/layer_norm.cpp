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

#define LBANN_LAYER_NORM_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/layer_norm.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/optimizers/optimizer_impl.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

namespace {

/** @brief Forward prop */
template <typename TensorDataType>
void fp_impl(lbann_comm& comm,
             TensorDataType epsilon,
             const El::AbstractDistMatrix<TensorDataType>& input,
             El::AbstractDistMatrix<TensorDataType>& output,
             El::AbstractDistMatrix<TensorDataType>& statistics,
             const TensorDataType* local_scale,
             const TensorDataType* local_bias)
{
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Workspace buffer
  statistics.Empty(false);
  statistics.AlignWith(input);
  statistics.Resize(2, input.Width());

  // Local matrices
  const auto& local_input =
    dynamic_cast<const CPUMatType&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<CPUMatType&>(output.Matrix());
  auto& local_statistics = dynamic_cast<CPUMatType&>(statistics.Matrix());
  auto local_means = El::LockedView(local_statistics, El::IR(0), El::ALL);
  auto local_vars = El::LockedView(local_statistics, El::IR(1), El::ALL);

  // Dimensions
  const El::Int sample_size = input.Height();
  const El::Int local_num_samples = local_input.Width();
  const El::Int local_sample_size = local_input.Height();

  // Compute sums
  El::Zero(statistics);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_num_samples; ++i) {
    auto& sum = local_means(0, i);
    auto& sqsum = local_vars(0, i);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j, i);
      sum += x;
      sqsum += x * x;
    }
  }
  comm.allreduce(statistics, statistics.RedundantComm(), El::mpi::SUM);

  // Compute statistics from sums
  //   mean = sum(x_i) / n
  //   var = ( sum(x_i^2)/n - mean^2 )
  if (sample_size <= 1) {
    // local_means already has correct values
    El::Fill(local_vars, El::TypeTraits<TensorDataType>::One());
  }
  else {
    LBANN_OMP_PARALLEL_FOR
    for (El::Int i = 0; i < local_num_samples; ++i) {
      const auto sum = local_means(0, i);
      const auto sqsum = local_vars(0, i);
      auto sample_size_dt = El::To<TensorDataType>(sample_size);
      const auto& mean = sum / sample_size_dt;
      const auto& sqmean = sqsum / sample_size_dt;
      const auto& var = (sqmean - mean * mean);
      local_means(0, i) = mean;
      local_vars(0, i) = std::max(var, El::TypeTraits<TensorDataType>::Zero());
    }
  }

  // Apply layer norm
  //   y_i = (x_i - mean) / sqrt(var + epsilon)
  for (El::Int i = 0; i < local_num_samples; ++i) {
    const auto& mean = local_means(0, i);
    const auto& var = local_vars(0, i);
    const TensorDataType inv_stdev =
      El::TypeTraits<TensorDataType>::One() / El::Sqrt(var + epsilon);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j, i);
      auto& y = local_output(j, i);
      TensorDataType result = (x - mean) * inv_stdev;
      if (local_scale)
        result *= local_scale[i];
      if (local_bias)
        result += local_bias[i];
      y = result;
    }
  }
}

/** @brief Backprop */
template <typename TensorDataType>
void bp_impl(lbann_comm& comm,
             TensorDataType epsilon,
             const El::AbstractDistMatrix<TensorDataType>& input,
             const El::AbstractDistMatrix<TensorDataType>& output_grad,
             El::AbstractDistMatrix<TensorDataType>& input_grad,
             const El::AbstractDistMatrix<TensorDataType>& statistics,
             El::AbstractDistMatrix<TensorDataType>& statistics_grad,
             const TensorDataType* local_scale,
             TensorDataType* scale_grad,
             TensorDataType* bias_grad)
{
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Workspace buffer
  statistics_grad.Empty(false);
  statistics_grad.AlignWith(input);
  statistics_grad.Resize(2, input.Width());

  // Local matrices
  const auto& local_input =
    dynamic_cast<const CPUMatType&>(input.LockedMatrix());
  const auto& local_output_grad =
    dynamic_cast<const CPUMatType&>(output_grad.LockedMatrix());
  auto& local_input_grad = dynamic_cast<CPUMatType&>(input_grad.Matrix());
  const auto& local_statistics =
    dynamic_cast<const CPUMatType&>(statistics.LockedMatrix());
  const auto local_means = El::LockedView(local_statistics, El::IR(0), El::ALL);
  const auto local_vars = El::LockedView(local_statistics, El::IR(1), El::ALL);
  auto& local_statistics_grad =
    dynamic_cast<CPUMatType&>(statistics_grad.Matrix());
  auto local_means_grad = El::View(local_statistics_grad, El::IR(0), El::ALL);
  auto local_vars_grad = El::View(local_statistics_grad, El::IR(1), El::ALL);

  // Dimensions
  const El::Int sample_size = input.Height();
  const El::Int local_num_samples = local_input.Width();
  const El::Int local_sample_size = local_input.Height();

  // Trivial case if sample size <= 1
  // Note: Output is constant, so error signal is zero.
  if (sample_size <= 1) {
    El::Zero(input_grad);
    return;
  }

  // Compute gradient w.r.t. statistics
  //   dL/dmean = - sum(dL/dy_i) / sqrt(var+epsilon)
  //   dL/dvar = - sum(dL/dy_i * (x_i-mean)) * (var+epsilon)^(-3/2) / 2
  El::Zero(statistics_grad);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_num_samples; ++i) {
    const auto& mean = local_means(0, i);
    const auto& var = local_vars(0, i);
    const TensorDataType inv_stdev =
      El::TypeTraits<TensorDataType>::One() / El::Sqrt(var + epsilon);
    auto& dmean = local_means_grad(0, i);
    auto& dvar = local_vars_grad(0, i);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j, i);
      auto dy = local_output_grad(j, i);

      if (bias_grad)
        bias_grad[i] = dy;
      if (scale_grad) {
        scale_grad[i] = dy * (x - mean) * inv_stdev;
        dy *= local_scale[i];
      }

      dmean += dy;
      dvar += dy * (x - mean);
    }
    dmean *= -inv_stdev;
    dvar *= -inv_stdev * inv_stdev * inv_stdev / 2;
  }
  comm.allreduce(statistics_grad,
                 statistics_grad.RedundantComm(),
                 El::mpi::SUM);

  // Compute gradient w.r.t. input
  //   dL/dx_i = ( dL/dy_i / sqrt(var+epsilon)
  //             + dL/dmean / n
  //             + dL/dvar * (x_i - mean) * 2/(n-1) )
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_num_samples; ++i) {
    const auto& mean = local_means(0, i);
    const auto& var = local_vars(0, i);
    const TensorDataType inv_stdev =
      El::TypeTraits<TensorDataType>::One() / El::Sqrt(var + epsilon);
    const auto& dmean = local_means_grad(0, i);
    const auto& dvar = local_vars_grad(0, i);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j, i);
      auto dy = local_output_grad(j, i);
      auto& dx = local_input_grad(j, i);

      if (local_scale)
        dy *= local_scale[i];

      dx = (dy * inv_stdev + dmean / sample_size +
            dvar * (x - mean) * 2 / sample_size);
    }
  }
}

} // namespace

// Template instantiation
template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType, Layout, Device>::fp_compute()
{
  int weight_idx = 0;
  const TensorDataType* scale_weights = nullptr;
  const TensorDataType* bias_weights = nullptr;
  if (m_scale)
    scale_weights =
      this->weights_values(weight_idx++).LockedMatrix().LockedBuffer();
  if (m_bias)
    bias_weights =
      this->weights_values(weight_idx).LockedMatrix().LockedBuffer();

  fp_impl(*this->get_comm(),
          this->m_epsilon,
          this->get_prev_activations(),
          this->get_activations(),
          *this->m_statistics,
          scale_weights,
          bias_weights);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType, Layout, Device>::bp_compute()
{
  // Obtain optional buffers
  const TensorDataType* scale_weights = nullptr;
  TensorDataType* scale_grad =
    m_scale ? this->m_scale_gradient->Buffer() : nullptr;
  TensorDataType* bias_grad =
    m_bias ? this->m_bias_gradient->Buffer() : nullptr;
  if (m_scale)
    scale_weights = this->weights_values(0).LockedMatrix().LockedBuffer();

  // Compute backpropagation
  bp_impl(*this->get_comm(),
          this->m_epsilon,
          this->get_prev_activations(),
          this->get_prev_error_signals(),
          this->get_error_signals(),
          *this->m_statistics,
          *this->m_statistics_gradient,
          scale_weights,
          scale_grad,
          bias_grad);

  // Update optimizers with gradients
  int weight_idx = 0;
  if (m_scale) {
    auto* opt = this->get_weights(weight_idx++).get_optimizer();
    if (opt != nullptr) {
      opt->add_to_gradient(*this->m_scale_gradient,
                           El::TypeTraits<TensorDataType>::One(),
                           true);
    }
  }
  if (m_bias) {
    auto* opt = this->get_weights(weight_idx).get_optimizer();
    if (opt != nullptr) {
      opt->add_to_gradient(*this->m_bias_gradient,
                           El::TypeTraits<TensorDataType>::One(),
                           true);
    }
  }
}

#define PROTO(T)                                                               \
  template class layer_norm_layer<T,                                           \
                                  data_layout::DATA_PARALLEL,                  \
                                  El::Device::CPU>;                            \
  template class layer_norm_layer<T,                                           \
                                  data_layout::MODEL_PARALLEL,                 \
                                  El::Device::CPU>

#include "lbann/macros/instantiate.hpp"

} // namespace lbann
