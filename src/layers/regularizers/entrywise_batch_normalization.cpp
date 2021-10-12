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

#define LBANN_ENTRYWISE_BATCH_NORMALIZATION_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/entrywise_batch_normalization.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"

#include "lbann/weights/weights_helpers.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

namespace {

// Block size for loops
// Note: x86 cache lines are 64B
constexpr El::Int _bsize = 64 / sizeof(DataType);
constexpr El::Int bsize = _bsize > 1 ? _bsize : 1;

/**
 *  mean = sum(x_i) / n
 *
 *  var = ( sum(x_i^2)/n - mean^2 ) * n/(n-1)
 */
template <typename TensorDataType>
void compute_batch_statistics(
  lbann_comm& comm,
  TensorDataType decay,
  const El::AbstractDistMatrix<TensorDataType>& input,
  El::AbstractDistMatrix<TensorDataType>& batch_statistics,
  El::AbstractDistMatrix<TensorDataType>& running_mean,
  El::AbstractDistMatrix<TensorDataType>& running_var)
{
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Local matrices
  const auto& local_input =
    dynamic_cast<const CPUMatType&>(input.LockedMatrix());
  auto& local_batch_statistics =
    dynamic_cast<CPUMatType&>(batch_statistics.Matrix());
  auto local_batch_mean = El::View(local_batch_statistics, El::ALL, El::IR(0));
  auto local_batch_var = El::View(local_batch_statistics, El::ALL, El::IR(1));
  auto& local_running_mean = dynamic_cast<CPUMatType&>(running_mean.Matrix());
  auto& local_running_var = dynamic_cast<CPUMatType&>(running_var.Matrix());

  // Dimensions
  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();

  // Compute local sums
  El::Zero(batch_statistics);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int row_start = 0; row_start < local_height; row_start += bsize) {
    const El::Int row_end = std::min(row_start + bsize, local_height);
    const El::Int col_start = 0;
    const El::Int col_end = local_width;
    for (El::Int col = col_start; col < col_end; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& x = local_input(row, col);
        local_batch_mean(row, 0) += x;
        local_batch_var(row, 0) += x * x;
      }
    }
  }

  // Accumulate sums between processes
  /// @todo Local statistics
  /// @todo Arbitrary group sizes
  comm.allreduce(batch_statistics,
                 batch_statistics.RedundantComm(),
                 El::mpi::SUM);
  const size_t statistics_count = input.Width();

  // Compute mini-batch statistics from sums
  if (statistics_count <= 1) {
    // local_mean already has correct values
    El::Fill(local_batch_var, El::TypeTraits<TensorDataType>::One());
  }
  else {
    LBANN_OMP_PARALLEL_FOR
    for (El::Int row = 0; row < local_height; ++row) {
      auto& mean = local_batch_mean(row, 0);
      auto& var = local_batch_var(row, 0);
      auto& _running_mean = local_running_mean(row, 0);
      auto& _running_var = local_running_var(row, 0);
      const auto sum = local_batch_mean(row, 0);
      const auto sqsum = local_batch_var(row, 0);
      mean = sum / statistics_count;
      const auto sqmean = sqsum / statistics_count;
      var = (sqmean - mean * mean) * statistics_count / (statistics_count - 1);
      _running_mean = decay * _running_mean + (DataType{1} - decay) * mean;
      _running_var = decay * _running_var + (DataType{1} - decay) * var;
    }
  }
}

/**
 *  y_i = (x_i - mean) / sqrt(var + epsilon)
 */
template <typename TensorDataType>
void apply_batchnorm(
  DataType epsilon,
  const El::Matrix<TensorDataType, El::Device::CPU>& local_input,
  El::Matrix<TensorDataType, El::Device::CPU>& local_output,
  const El::Matrix<TensorDataType, El::Device::CPU>& local_mean,
  const El::Matrix<TensorDataType, El::Device::CPU>& local_var)
{
  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();
  LBANN_OMP_PARALLEL_FOR
  for (El::Int row_start = 0; row_start < local_height; row_start += bsize) {
    const El::Int row_end = std::min(row_start + bsize, local_height);
    const El::Int col_start = 0;
    const El::Int col_end = local_width;
    TensorDataType _inv_stdev[bsize];
    for (El::Int row = row_start; row < row_end; ++row) {
      const auto& var = local_var(row, 0);
      _inv_stdev[row - row_start] = 1 / El::Sqrt(var + epsilon);
    }
    for (El::Int col = col_start; col < col_end; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& mean = local_mean(row, 0);
        const auto& inv_stdev = _inv_stdev[row - row_start];
        const auto& x = local_input(row, col);
        auto& y = local_output(row, col);
        y = (x - mean) * inv_stdev;
      }
    }
  }
}

template <typename TensorDataType>
void fp_impl(lbann_comm& comm,
             TensorDataType decay,
             TensorDataType epsilon,
             bool is_training,
             const El::AbstractDistMatrix<TensorDataType>& input,
             El::AbstractDistMatrix<TensorDataType>& output,
             El::AbstractDistMatrix<TensorDataType>& batch_statistics,
             El::AbstractDistMatrix<TensorDataType>& running_mean,
             El::AbstractDistMatrix<TensorDataType>& running_var)
{
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Make sure workspace is aligned with input tensor
  batch_statistics.Empty(false);
  batch_statistics.AlignWith(input);
  batch_statistics.Resize(input.Height(), 2);

  // Local matrices
  const auto& local_input =
    dynamic_cast<const CPUMatType&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<CPUMatType&>(output.Matrix());

  // Batchnorm has different behavior for training and inference
  if (is_training) {

    // For training, normalize with batch statistics
    compute_batch_statistics<TensorDataType>(comm,
                                             decay,
                                             input,
                                             batch_statistics,
                                             running_mean,
                                             running_var);
    const auto& local_batch_statistics =
      dynamic_cast<const CPUMatType&>(batch_statistics.LockedMatrix());
    const auto local_batch_mean =
      El::LockedView(local_batch_statistics, El::ALL, El::IR(0));
    const auto local_batch_var =
      El::LockedView(local_batch_statistics, El::ALL, El::IR(1));
    apply_batchnorm<TensorDataType>(epsilon,
                                    local_input,
                                    local_output,
                                    local_batch_mean,
                                    local_batch_var);
  }
  else {

    // For inference, normalize with running statistics
    const auto& local_running_mean =
      dynamic_cast<const CPUMatType&>(running_mean.LockedMatrix());
    const auto& local_running_var =
      dynamic_cast<const CPUMatType&>(running_var.LockedMatrix());
    apply_batchnorm<TensorDataType>(epsilon,
                                    local_input,
                                    local_output,
                                    local_running_mean,
                                    local_running_var);
  }
}

/** @brief Backprop for training.
 *
 *  Assumes forward prop uses mini-batch statistics. In other words,
 *  statistics are dependent on input.
 */
template <typename TensorDataType>
void bp_training_impl(
  lbann_comm& comm,
  TensorDataType epsilon,
  const El::AbstractDistMatrix<TensorDataType>& input,
  const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
  El::AbstractDistMatrix<TensorDataType>& gradient_wrt_input,
  const El::AbstractDistMatrix<TensorDataType>& statistics,
  El::AbstractDistMatrix<TensorDataType>& gradient_wrt_statistics)
{
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Make sure workspace is aligned with input tensor
  gradient_wrt_statistics.Empty(false);
  gradient_wrt_statistics.AlignWith(input);
  gradient_wrt_statistics.Resize(input.Height(), 2);

  // Local matrices
  const auto& local_input =
    dynamic_cast<const CPUMatType&>(input.LockedMatrix());
  const auto& local_gradient_wrt_output =
    dynamic_cast<const CPUMatType&>(gradient_wrt_output.LockedMatrix());
  auto& local_gradient_wrt_input =
    dynamic_cast<CPUMatType&>(gradient_wrt_input.Matrix());
  const auto& local_statistics =
    dynamic_cast<const CPUMatType&>(statistics.LockedMatrix());
  const auto local_mean = El::LockedView(local_statistics, El::ALL, El::IR(0));
  const auto local_var = El::LockedView(local_statistics, El::ALL, El::IR(1));
  auto& local_gradient_wrt_statistics =
    dynamic_cast<CPUMatType&>(gradient_wrt_statistics.Matrix());
  auto local_gradient_wrt_mean =
    El::View(local_gradient_wrt_statistics, El::ALL, El::IR(0));
  auto local_gradient_wrt_var =
    El::View(local_gradient_wrt_statistics, El::ALL, El::IR(1));

  // Dimensions
  const El::Int local_height = local_gradient_wrt_input.Height();
  const El::Int local_width = local_gradient_wrt_input.Width();

  // Count for statistics
  // Note: Output is constant if statistics count is <=1, so error
  // signal is zero.
  /// @todo Local statistics
  /// @todo Arbitrary group sizes
  const size_t statistics_count = input.Width();
  if (statistics_count <= 1) {
    El::Zero(local_gradient_wrt_input);
    return;
  }

  // Compute local gradient w.r.t. batch statistics
  //   dL/dmean = - sum(dL/dy_i) / sqrt(var+epsilon)
  //   dL/dvar = - sum(dL/dy_i * (x_i-mean)) * (var+epsilon)^(-3/2) / 2
  El::Zero(gradient_wrt_statistics);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int row_start = 0; row_start < local_height; row_start += bsize) {
    const El::Int row_end = std::min(row_start + bsize, local_height);
    const El::Int col_start = 0;
    const El::Int col_end = local_width;
    TensorDataType _inv_stdev[bsize];
    for (El::Int row = row_start; row < row_end; ++row) {
      const auto& var = local_var(row, 0);
      _inv_stdev[row - row_start] = 1 / El::Sqrt(var + epsilon);
    }
    for (El::Int col = col_start; col < col_end; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& mean = local_mean(row, 0);
        const auto& inv_stdev = _inv_stdev[row - row_start];
        const auto& x = local_input(row, col);
        const auto& dy = local_gradient_wrt_output(row, col);
        auto& dmean = local_gradient_wrt_mean(row, 0);
        auto& dvar = local_gradient_wrt_var(row, 0);
        dmean += -dy * inv_stdev;
        dvar += -dy * (x - mean) * inv_stdev * inv_stdev * inv_stdev / 2;
      }
    }
  }

  // Accumulate gradient w.r.t. statistics across processes
  /// @todo Local statistics
  /// @todo Arbitrary group sizes
  comm.allreduce(gradient_wrt_statistics,
                 gradient_wrt_statistics.RedundantComm(),
                 El::mpi::SUM);

  // Compute gradient w.r.t. input
  //   dL/dx_i = ( dL/dy_i / sqrt(var+epsilon)
  //             + dL/dmean / n
  //             + dL/dvar * (x_i - mean) * 2/(n-1) )
  const auto statistics_count_dt = El::To<TensorDataType>(statistics_count);
  const TensorDataType inv_stats_count =
    El::TypeTraits<TensorDataType>::One() / statistics_count_dt;
  const TensorDataType inv_stats_countm1 =
    El::TypeTraits<TensorDataType>::One() /
    (statistics_count_dt - El::TypeTraits<TensorDataType>::One());
  LBANN_OMP_PARALLEL_FOR
  for (El::Int row_start = 0; row_start < local_height; row_start += bsize) {
    const El::Int row_end = std::min(row_start + bsize, local_height);
    const El::Int col_start = 0;
    const El::Int col_end = local_width;
    TensorDataType _inv_stdev[bsize];
    for (El::Int row = row_start; row < row_end; ++row) {
      const auto& var = local_var(row, 0);
      _inv_stdev[row - row_start] = 1 / El::Sqrt(var + epsilon);
    }
    for (El::Int col = col_start; col < col_end; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& mean = local_mean(row, 0);
        const auto& inv_stdev = _inv_stdev[row - row_start];
        const auto& x = local_input(row, col);
        const auto& dy = local_gradient_wrt_output(row, col);
        auto& dx = local_gradient_wrt_input(row, col);
        auto& dmean = local_gradient_wrt_mean(row, 0);
        auto& dvar = local_gradient_wrt_var(row, 0);
        dx = (dy * inv_stdev + dmean * inv_stats_count +
              dvar * (x - mean) * 2 * inv_stats_countm1);
      }
    }
  }
}

/** @brief Backprop for inference.
 *
 *  Computes gradient w.r.t. input when the model is performing
 *  inference, e.g. in validation or testing mode. In this case,
 *  forward prop uses running statistics, which are independent of
 *  input.
 */
template <typename TensorDataType>
void bp_inference_impl(
  DataType epsilon,
  const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
  El::AbstractDistMatrix<TensorDataType>& gradient_wrt_input,
  const El::AbstractDistMatrix<TensorDataType>& running_var)
{
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Local matrices
  const auto& local_gradient_wrt_output =
    dynamic_cast<const CPUMatType&>(gradient_wrt_output.LockedMatrix());
  auto& local_gradient_wrt_input =
    dynamic_cast<CPUMatType&>(gradient_wrt_input.Matrix());
  const auto& local_running_var =
    dynamic_cast<const CPUMatType&>(running_var.LockedMatrix());

  // Compute gradient w.r.t. input
  //   dL/dx_i = dL/dy_i / sqrt(var+epsilon)
  const El::Int local_height = local_gradient_wrt_input.Height();
  const El::Int local_width = local_gradient_wrt_input.Width();
  LBANN_OMP_PARALLEL_FOR
  for (El::Int row_start = 0; row_start < local_height; row_start += bsize) {
    const El::Int row_end = std::min(row_start + bsize, local_height);
    const El::Int col_start = 0;
    const El::Int col_end = local_width;
    TensorDataType _inv_stdev[bsize];
    for (El::Int row = row_start; row < row_end; ++row) {
      const auto& var = local_running_var(row, 0);
      _inv_stdev[row - row_start] = 1 / El::Sqrt(var + epsilon);
    }
    for (El::Int col = col_start; col < col_end; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& inv_stdev = _inv_stdev[row - row_start];
        const auto& dy = local_gradient_wrt_output(row, col);
        auto& dx = local_gradient_wrt_input(row, col);
        dx = dy * inv_stdev;
      }
    }
  }
}

template <typename TensorDataType>
void bp_impl(
  lbann_comm& comm,
  TensorDataType epsilon,
  bool is_training,
  const El::AbstractDistMatrix<TensorDataType>& input,
  const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
  El::AbstractDistMatrix<TensorDataType>& gradient_wrt_input,
  const El::AbstractDistMatrix<TensorDataType>& batch_statistics,
  El::AbstractDistMatrix<TensorDataType>& gradient_wrt_batch_statistics,
  const El::AbstractDistMatrix<TensorDataType>& running_var)
{

  // Batchnorm has different behavior for training and inference
  if (is_training) {
    bp_training_impl<TensorDataType>(comm,
                                     epsilon,
                                     input,
                                     gradient_wrt_output,
                                     gradient_wrt_input,
                                     batch_statistics,
                                     gradient_wrt_batch_statistics);
  }
  else {
    bp_inference_impl<TensorDataType>(epsilon,
                                      gradient_wrt_output,
                                      gradient_wrt_input,
                                      running_var);
  }
}

} // namespace

// Template instantiation
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void entrywise_batch_normalization_layer<TensorDataType, T_layout, Dev>::
  fp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("entrywise_batch_normalization_layer::fp_compute");
  using ValuesGetter = weights_details::SafeWeightsAccessor<TensorDataType>;

  const auto mode = this->m_model->get_execution_context().get_execution_mode();
  fp_impl(*this->get_comm(),
          this->m_decay,
          this->m_epsilon,
          mode == execution_mode::training,
          this->get_prev_activations(),
          this->get_activations(),
          *this->m_batch_statistics,
          ValuesGetter::mutable_values(this->get_weights(0)),
          ValuesGetter::mutable_values(this->get_weights(1)));
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void entrywise_batch_normalization_layer<TensorDataType, T_layout, Dev>::
  bp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("entrywise_batch_normalization_layer::bp_compute");
  const auto mode = this->m_model->get_execution_context().get_execution_mode();
  bp_impl(*this->get_comm(),
          this->m_epsilon,
          mode == execution_mode::training,
          this->get_prev_activations(),
          this->get_prev_error_signals(),
          this->get_error_signals(),
          *this->m_batch_statistics,
          *this->m_batch_statistics_gradient,
          this->weights_values(1));
}

#define PROTO(T)                                                               \
  template class entrywise_batch_normalization_layer<                          \
    T,                                                                         \
    data_layout::DATA_PARALLEL,                                                \
    El::Device::CPU>;                                                          \
  template class entrywise_batch_normalization_layer<                          \
    T,                                                                         \
    data_layout::MODEL_PARALLEL,                                               \
    El::Device::CPU>

#include "lbann/macros/instantiate.hpp"

} // namespace lbann
