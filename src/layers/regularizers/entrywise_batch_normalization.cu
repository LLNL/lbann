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

#define LBANN_ENTRYWISE_BATCH_NORMALIZATION_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/entrywise_batch_normalization.hpp"
#include "lbann/weights/weights_helpers.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

/**
 *  On input, sums and sqsums are assumed to be filled with zeros.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (height / bsize) x 1 x 1
 */
template <typename TensorDataType>
__global__ void row_sums_kernel(size_t height,
                                size_t width,
                                const TensorDataType* __restrict__ vals,
                                size_t vals_ldim,
                                TensorDataType* __restrict__ sums,
                                TensorDataType* __restrict__ sqsums) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  for (size_t row = gid; row < height; row += nthreads) {
    auto& sum = sums[row];
    auto& sqsum = sqsums[row];
    for (size_t col = 0; col < width; ++col) {
      const auto& x = vals[row + col * vals_ldim];
      sum += x;
      sqsum += x * x;
    }
  }
}

/**
 *  On input, batch_mean and batch_var are assumed to contain sums and
 *  squares of sums, respectively.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (size / bsize) x 1 x 1
 */
template <typename TensorDataType>
__global__ void compute_statistics_kernel(size_t size,
                                          unsigned long long statistics_count,
                                          TensorDataType decay,
                                          TensorDataType* __restrict__ batch_mean,
                                          TensorDataType* __restrict__ batch_var,
                                          TensorDataType* __restrict__ running_mean,
                                          TensorDataType* __restrict__ running_var) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  for (size_t i = gid; i < size; i += nthreads) {
    auto& mean = batch_mean[i];
    auto& var = batch_var[i];
    auto& _running_mean = running_mean[i];
    auto& _running_var = running_var[i];
    const auto sum = batch_mean[i];
    const auto sqsum = batch_var[i];
    const TensorDataType statistics_count_dt = TensorDataType(statistics_count);
    mean = sum / statistics_count_dt;
    const auto sqmean = sqsum / statistics_count_dt;
    var = (sqmean - mean * mean) * statistics_count_dt / TensorDataType(statistics_count - 1);
    _running_mean = decay * _running_mean + (TensorDataType{1} - decay) * mean;
    _running_var = decay * _running_var + (TensorDataType{1} - decay) * var;
  }
}

/**
 *  mean = sum(x_i) / n
 *
 *  var = ( sum(x_i^2)/n - mean^2 ) * n/(n-1)
 */
template <typename TensorDataType>
void compute_batch_statistics(lbann_comm& comm,
                              TensorDataType decay,
                              const El::AbstractDistMatrix<TensorDataType>& input,
                              El::AbstractDistMatrix<TensorDataType>& batch_statistics,
                              El::AbstractDistMatrix<TensorDataType>& running_mean,
                              El::AbstractDistMatrix<TensorDataType>& running_var) {

  // Local matrices
  const auto& local_input = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(input.LockedMatrix());
  auto& local_batch_statistics = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(batch_statistics.Matrix());
  auto local_batch_mean = El::View(local_batch_statistics, El::ALL, El::IR(0));
  auto local_batch_var = El::View(local_batch_statistics, El::ALL, El::IR(1));
  auto& local_running_mean = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(running_mean.Matrix());
  auto& local_running_var = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(running_var.Matrix());

  // Dimensions
  const size_t local_height = local_input.Height();
  const size_t local_width = local_input.Width();

  // Compute local sums
  El::Zero(batch_statistics);
  if (local_height > 0) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_batch_statistics),
                        gpu::get_sync_info(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    hydrogen::gpu::LaunchKernel(
      row_sums_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_height,
      local_width,
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_batch_mean.Buffer(),
      local_batch_var.Buffer());
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
  } else {
    if (local_height > 0) {
      auto multisync =
        El::MakeMultiSync(gpu::get_sync_info(local_batch_statistics),
                          gpu::get_sync_info(local_running_mean),
                          gpu::get_sync_info(local_running_var));

      constexpr size_t block_size = 256;
      dim3 block_dims, grid_dims;
      block_dims.x = block_size;
      grid_dims.x = (local_height + block_size - 1) / block_size;
      hydrogen::gpu::LaunchKernel(
        compute_statistics_kernel<TensorDataType>,
        grid_dims, block_dims, 0, multisync,
        local_height,
        statistics_count,
        decay,
        local_batch_mean.Buffer(),
        local_batch_var.Buffer(),
        local_running_mean.Buffer(),
        local_running_var.Buffer());
    }
  }

}

/**
 *  Block dimensions: bsizex x bsizey x 1
 *
 *  Grid dimensions: (height / bsizex) x (width / bsizey) x 1
 */
template <typename TensorDataType>
__global__ void batchnorm_kernel(size_t height,
                                 size_t width,
                                 TensorDataType epsilon,
                                 const TensorDataType* __restrict__ input,
                                 size_t input_ldim,
                                 TensorDataType* __restrict__ output,
                                 size_t output_ldim,
                                 const TensorDataType* __restrict__ mean,
                                 const TensorDataType* __restrict__ var) {
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t row = gidx; row < height; row += nthreadsx) {
    const auto& _mean = mean[row];
    const auto& _var = var[row];
    const auto inv_stdev = cuda::rsqrt(_var + epsilon);
    for (size_t col = gidy; col < width; col += nthreadsy) {
      const auto& x = input[row + col*input_ldim];
      auto& y = output[row + col*output_ldim];
      y = (x - _mean) * inv_stdev;
    }
  }
}

/**
 *  y_i = (x_i - mean) / sqrt(var + epsilon)
 */
template <typename TensorDataType>
void apply_batchnorm(DataType epsilon,
                     const El::Matrix<TensorDataType, El::Device::GPU>& local_input,
                     El::Matrix<TensorDataType, El::Device::GPU>& local_output,
                     const El::Matrix<TensorDataType, El::Device::GPU>& local_mean,
                     const El::Matrix<TensorDataType, El::Device::GPU>& local_var) {
  if (!local_input.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_output),
                                       gpu::get_sync_info(local_input),
                                       gpu::get_sync_info(local_mean),
                                       gpu::get_sync_info(local_var));
    const size_t local_height = local_input.Height();
    const size_t local_width = local_input.Width();
    constexpr size_t block_size_x = 256;
    constexpr size_t block_size_y = 1;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size_x;
    block_dims.y = block_size_y;
    grid_dims.x = (local_height + block_size_x - 1) / block_size_x;
    grid_dims.y = (local_width + block_size_y - 1) / block_size_y;
    hydrogen::gpu::LaunchKernel(
      batchnorm_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_height,
      local_width,
      epsilon,
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_output.Buffer(),
      local_output.LDim(),
      local_mean.LockedBuffer(),
      local_var.LockedBuffer());
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
             El::AbstractDistMatrix<TensorDataType>& running_var) {

  // Local matrices
  const auto& local_input = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(output.Matrix());

  // Batchnorm has different behavior for training and inference
  if (is_training) {

    // For training, normalize with batch statistics
    compute_batch_statistics<TensorDataType>(comm,
                                             decay,
                                             input,
                                             batch_statistics,
                                             running_mean,
                                             running_var);
    const auto& local_batch_statistics
      = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(batch_statistics.LockedMatrix());
    const auto local_batch_mean = El::LockedView(local_batch_statistics,
                                                 El::ALL, El::IR(0));
    const auto local_batch_var = El::LockedView(local_batch_statistics,
                                                El::ALL, El::IR(1));
    apply_batchnorm<TensorDataType>(epsilon,
                                    local_input,
                                    local_output,
                                    local_batch_mean,
                                    local_batch_var);

  }
  else {

    // For inference, normalize with running statistics
    const auto& local_running_mean = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(running_mean.LockedMatrix());
    const auto& local_running_var = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(running_var.LockedMatrix());
    apply_batchnorm<TensorDataType>(epsilon,
                                    local_input,
                                    local_output,
                                    local_running_mean,
                                    local_running_var);

  }

}

/**
 *  On input, gradient_wrt_mean and gradient_wrt_var are assumed to be
 *  filled with zeros.
 *
 *  dL/dmean = - sum(dL/dy_i) / sqrt(var+epsilon)
 *
 *  dL/dvar = - sum(dL/dy_i * (x_i-mean)) * (var+epsilon)^(-3/2) / 2
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (height / bsize) x 1 x 1
 */
template <typename TensorDataType>
__global__ void bp_training_stats_gradient_kernel(size_t height,
                                                  size_t width,
                                                  TensorDataType epsilon,
                                                  const TensorDataType* __restrict__ input,
                                                  size_t input_ldim,
                                                  const TensorDataType* __restrict__ gradient_wrt_output,
                                                  size_t gradient_wrt_output_ldim,
                                                  const TensorDataType* __restrict__ mean,
                                                  const TensorDataType* __restrict__ var,
                                                  TensorDataType* __restrict__ gradient_wrt_mean,
                                                  TensorDataType* __restrict__ gradient_wrt_var) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  for (size_t row = gid; row < height; row += nthreads) {
    const auto& _mean = mean[row];
    const auto& _var = var[row];
    const auto inv_stdev = cuda::rsqrt(_var + epsilon);
    auto& dmean = gradient_wrt_mean[row];
    auto& dvar = gradient_wrt_var[row];
    for (size_t col = 0; col < width; ++col) {
      const auto& x = input[row + col * input_ldim];
      const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      dmean += - dy * inv_stdev;
      dvar += - dy * (x - _mean) * inv_stdev*inv_stdev*inv_stdev / TensorDataType(2);
    }
  }
}

/**
 *  dL/dx_i = ( dL/dy_i / sqrt(var+epsilon)
 *              + dL/dmean / n
 *              + dL/dvar * (x_i - mean) * 2/(n-1) )
 *
 *  Block dimensions: bsizex x bsizey x 1
 *
 *  Grid dimensions: (height / bsizex) x (width / bsizey) x 1
 */
template <typename TensorDataType>
__global__ void bp_training_error_signal_kernel(size_t height,
                                                size_t width,
                                                TensorDataType epsilon,
                                                unsigned long long statistics_count,
                                                const TensorDataType* __restrict__ input,
                                                size_t input_ldim,
                                                const TensorDataType* __restrict__ gradient_wrt_output,
                                                size_t gradient_wrt_output_ldim,
                                                TensorDataType* __restrict__ gradient_wrt_input,
                                                size_t gradient_wrt_input_ldim,
                                                const TensorDataType* __restrict__ mean,
                                                const TensorDataType* __restrict__ var,
                                                const TensorDataType* __restrict__ gradient_wrt_mean,
                                                const TensorDataType* __restrict__ gradient_wrt_var) {
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t row = gidx; row < height; row += nthreadsx) {
    const auto& _mean = mean[row];
    const auto& _var = var[row];
    const auto& dmean = gradient_wrt_mean[row];
    const auto& dvar = gradient_wrt_var[row];
    const auto inv_stdev = cuda::rsqrt(_var + epsilon);
    for (size_t col = gidy; col < width; col += nthreadsy) {
      const auto& x = input[row + col * input_ldim];
      const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
      dx = (dy * inv_stdev
            + dmean / TensorDataType(statistics_count)
            + dvar * (x - _mean) * TensorDataType(2) / TensorDataType(statistics_count - 1));
    }
  }
}

/** @brief Backprop for training.
 *
 *  Assumes forward prop uses mini-batch statistics. In other words,
 *  statistics are dependent on input.
 */
template <typename TensorDataType>
void bp_training_impl(lbann_comm& comm,
                      TensorDataType epsilon,
                      const El::AbstractDistMatrix<TensorDataType>& input,
                      const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
                      El::AbstractDistMatrix<TensorDataType>& gradient_wrt_input,
                      const El::AbstractDistMatrix<TensorDataType>& statistics,
                      El::AbstractDistMatrix<TensorDataType>& gradient_wrt_statistics) {

  // Local matrices
  const auto& local_input = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(input.LockedMatrix());
  const auto& local_gradient_wrt_output = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(gradient_wrt_output.LockedMatrix());
  auto& local_gradient_wrt_input = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(gradient_wrt_input.Matrix());
  const auto& local_statistics = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(statistics.LockedMatrix());
  const auto local_mean = El::LockedView(local_statistics, El::ALL, El::IR(0));
  const auto local_var = El::LockedView(local_statistics, El::ALL, El::IR(1));
  auto& local_gradient_wrt_statistics = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(gradient_wrt_statistics.Matrix());
  auto local_gradient_wrt_mean = El::View(local_gradient_wrt_statistics, El::ALL, El::IR(0));
  auto local_gradient_wrt_var = El::View(local_gradient_wrt_statistics, El::ALL, El::IR(1));

  // Dimensions
  const size_t local_height = local_gradient_wrt_input.Height();
  const size_t local_width = local_gradient_wrt_input.Width();

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
  El::Zero(gradient_wrt_statistics);
  if (local_height > 0) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_gradient_wrt_statistics),
                        gpu::get_sync_info(local_statistics),
                        gpu::get_sync_info(local_gradient_wrt_output),
                        gpu::get_sync_info(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    hydrogen::gpu::LaunchKernel(
      bp_training_stats_gradient_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_height,
      local_width,
      epsilon,
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_gradient_wrt_output.LockedBuffer(),
      local_gradient_wrt_output.LDim(),
      local_mean.LockedBuffer(),
      local_var.LockedBuffer(),
      local_gradient_wrt_mean.Buffer(),
      local_gradient_wrt_var.Buffer());
  }

  // Accumulate gradient w.r.t. statistics across processes
  /// @todo Local statistics
  /// @todo Arbitrary group sizes
  comm.allreduce(gradient_wrt_statistics,
                 gradient_wrt_statistics.RedundantComm(),
                 El::mpi::SUM);

  // Compute gradient w.r.t. input
  if (!local_input.IsEmpty()) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_gradient_wrt_input),
                        gpu::get_sync_info(local_gradient_wrt_statistics),
                        gpu::get_sync_info(local_statistics),
                        gpu::get_sync_info(local_gradient_wrt_output),
                        gpu::get_sync_info(local_input));
    const size_t local_height = local_input.Height();
    const size_t local_width = local_input.Width();
    constexpr size_t block_size_x = 256;
    constexpr size_t block_size_y = 1;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size_x;
    block_dims.y = block_size_y;
    grid_dims.x = (local_height + block_size_x - 1) / block_size_x;
    grid_dims.y = (local_width + block_size_y - 1) / block_size_y;
    hydrogen::gpu::LaunchKernel(
      bp_training_error_signal_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_height,
      local_width,
      epsilon,
      statistics_count,
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_gradient_wrt_output.LockedBuffer(),
      local_gradient_wrt_output.LDim(),
      local_gradient_wrt_input.Buffer(),
      local_gradient_wrt_input.LDim(),
      local_mean.LockedBuffer(),
      local_var.LockedBuffer(),
      local_gradient_wrt_mean.LockedBuffer(),
      local_gradient_wrt_var.LockedBuffer());
  }

}

/**
 *  dL/dx_i = dL/dy_i / sqrt(var+epsilon)
 *
 *  Block dimensions: bsizex x bsizey x 1
 *
 *  Grid dimensions: (height / bsizex) x (width / bsizey) x 1
 */
template <typename TensorDataType>
__global__ void bp_inference_kernel(size_t height,
                                    size_t width,
                                    TensorDataType epsilon,
                                    const TensorDataType* __restrict__ gradient_wrt_output,
                                    size_t gradient_wrt_output_ldim,
                                    TensorDataType* __restrict__ gradient_wrt_input,
                                    size_t gradient_wrt_input_ldim,
                                    const TensorDataType* __restrict__ running_var) {
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t row = gidx; row < height; row += nthreadsx) {
    const auto& var = running_var[row];
    const auto inv_stdev = cuda::rsqrt(var + epsilon);
    for (size_t col = gidy; col < width; col += nthreadsy) {
      const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
      dx = dy * inv_stdev;
    }
  }
}

/** @brief Backprop for inference.
 *
 *  Assumes forward prop uses running statistics. In other words,
 *  statistics are independent of input.
 */
template <typename TensorDataType>
void bp_inference_impl(DataType epsilon,
                       const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
                       El::AbstractDistMatrix<TensorDataType>& gradient_wrt_input,
                       const El::AbstractDistMatrix<TensorDataType>& running_var) {

  // Local matrices
  const auto& local_gradient_wrt_output = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(gradient_wrt_output.LockedMatrix());
  auto& local_gradient_wrt_input = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(gradient_wrt_input.Matrix());
  const auto& local_running_var = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(running_var.LockedMatrix());

  // Compute gradient w.r.t. input
  if (!local_gradient_wrt_output.IsEmpty()) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_gradient_wrt_input),
                        gpu::get_sync_info(local_gradient_wrt_output),
                        gpu::get_sync_info(local_running_var));
    const size_t local_height = local_gradient_wrt_output.Height();
    const size_t local_width = local_gradient_wrt_output.Width();
    constexpr size_t block_size_x = 256;
    constexpr size_t block_size_y = 1;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size_x;
    block_dims.y = block_size_y;
    grid_dims.x = (local_height + block_size_x - 1) / block_size_x;
    grid_dims.y = (local_width + block_size_y - 1) / block_size_y;
    hydrogen::gpu::LaunchKernel(
      bp_inference_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
        local_height,
        local_width,
        epsilon,
        local_gradient_wrt_output.LockedBuffer(),
        local_gradient_wrt_output.LDim(),
        local_gradient_wrt_input.Buffer(),
        local_gradient_wrt_input.LDim(),
        local_running_var.LockedBuffer());
  }

}

template <typename TensorDataType>
void bp_impl(lbann_comm& comm,
             TensorDataType epsilon,
             bool is_training,
             const El::AbstractDistMatrix<TensorDataType>& input,
             const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
             El::AbstractDistMatrix<TensorDataType>& gradient_wrt_input,
             const El::AbstractDistMatrix<TensorDataType>& batch_statistics,
             El::AbstractDistMatrix<TensorDataType>& gradient_wrt_batch_statistics,
             const El::AbstractDistMatrix<TensorDataType>& running_var) {

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
void entrywise_batch_normalization_layer<TensorDataType, T_layout, Dev>::fp_compute() {
  using ValuesGetter = weights_details::SafeWeightsAccessor<TensorDataType>;

  const auto mode = this->get_model()->get_execution_context().get_execution_mode();
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
void entrywise_batch_normalization_layer<TensorDataType, T_layout, Dev>::bp_compute() {
  const auto mode = this->get_model()->get_execution_context().get_execution_mode();
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

#define PROTO(T)                                      \
  template class entrywise_batch_normalization_layer< \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;  \
  template class entrywise_batch_normalization_layer< \
    T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
