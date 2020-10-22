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

#define LBANN_INSTANCE_NORM_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/instance_norm.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include <thrust/pair.h>

namespace lbann
{

namespace
{

/** Functor for adding @c thrust::pair objects. */
template <typename Pair>
struct pair_sum
{
  __device__ __forceinline__
  Pair operator()(const Pair& x, const Pair& y)
  {
    return Pair(x.first+y.first, x.second+y.second);
  }
};

} // namespace <anon>

// =============================================
// Forward prop
// =============================================

namespace
{

/** Accumulate sums and sums of squares for each channel.
 *
 *  On input, sums and sqsums are filled with zeros.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (channel_size / bsize) x num_channels x mini_batch_size
 */
template <typename TensorDataType, size_t bdimx>
__global__ void fp_sums_kernel(
  size_t mini_batch_size,
  size_t num_channels,
  size_t channel_size,
  const TensorDataType* __restrict__ vals,
  size_t vals_ldim,
  TensorDataType* sums,
  size_t sums_ldim,
  TensorDataType* sqsums,
  size_t sqsums_ldim)
{

  // Indices and dimensions
  constexpr size_t bdimy = 1;
  constexpr size_t bdimz = 1;
  const size_t tid = threadIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;

  for (size_t k = gidz; k < mini_batch_size; k += nthreadsz) {
    for (size_t j = gidy; j < num_channels; j += nthreadsy) {

      // Accumulate sums and perform block-wide reduction
      using pair_t = thrust::pair<TensorDataType,TensorDataType>;
      using pair_sum_t = pair_sum<pair_t>;
      pair_t sum_sqsum(0,0);
      for (size_t i = gidx; i < channel_size; i += nthreadsx) {
        const auto& x = vals[i + j*channel_size + k*vals_ldim];
        sum_sqsum.first += x;
        sum_sqsum.second += x * x;
      }
      sum_sqsum = gpu_lib::block_reduce<bdimx,bdimy,bdimz,pair_t,pair_sum_t>(sum_sqsum);

      // Output result to global memory
      if (tid == 0) {
        gpu_lib::atomic_add(&sums[j+k*sums_ldim], sum_sqsum.first);
        gpu_lib::atomic_add(&sqsums[j+k*sqsums_ldim], sum_sqsum.second);
      }

    }
  }

}

/** Compute outputs.
 *
 *  y_i = (x_i - mean) / sqrt(var + epsilon)
 *
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (channel_size / bdimx) x (num_channels / bdimy) x (mini_batch_size / bdimz)
 */
template <typename TensorDataType>
__global__ void fp_output_kernel(
  size_t mini_batch_size,
  size_t num_channels,
  size_t channel_size,
  TensorDataType epsilon,
  const TensorDataType* __restrict__ input,
  size_t input_ldim,
  TensorDataType* __restrict__ output,
  size_t output_ldim,
  const TensorDataType* sums,
  size_t sums_ldim,
  const TensorDataType* sqsums,
  size_t sqsums_ldim)
{

  // Indices
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;

  const TensorDataType mean_scale = 1. / channel_size;
  const TensorDataType var_correction = double(channel_size) / (channel_size - 1);
  for (size_t k = gidz; k < mini_batch_size; k += nthreadsz) {
    for (size_t j = gidy; j < num_channels; j += nthreadsy) {
      const auto& sum = sums[j+k*sums_ldim];
      const auto& sqsum = sqsums[j+k*sqsums_ldim];
      const auto& mean = sum * mean_scale;
      const auto& sqmean = sqsum * mean_scale;
      auto var = (sqmean - mean*mean) * var_correction;
      var = gpu_lib::max(var, TensorDataType{0.});
      const auto& inv_stdev = gpu_lib::rsqrt(var + epsilon);
      for (size_t i = gidx; i < channel_size; i += nthreadsx) {
        const auto& x = input[i + j*channel_size + k*input_ldim];
        auto& y = output[i + j*channel_size + k*output_ldim];
        y = (x - mean) * inv_stdev;
      }
    }
  }

}

/** @brief Forward prop */
template <typename TensorDataType>
void fp_impl(lbann_comm& comm,
             size_t num_channels,
             size_t channel_size,
             TensorDataType epsilon,
             const El::AbstractDistMatrix<TensorDataType>& input,
             El::AbstractDistMatrix<TensorDataType>& output,
             El::Matrix<TensorDataType, El::Device::GPU>& local_workspace)
{

  // Local matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<LocalMat&>(output.Matrix());

  // Dimensions
  const size_t local_mini_batch_size = local_input.Width();

  // Trivial case if channel size is 1
  // Note: Output is constant.
  if (channel_size <= 1) {
    El::Zero(output);
    return;
  }

  // Compute sums
  El::Zeros(local_workspace, 2*num_channels, local_mini_batch_size);
  auto local_sums = El::View(local_workspace,
                             El::IR(0, num_channels),
                             El::ALL);
  auto local_sqsums = El::View(local_workspace,
                               El::IR(num_channels, 2*num_channels),
                               El::ALL);
  if (!local_input.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_workspace),
                                       gpu::get_sync_info(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_mini_batch_size;
    hydrogen::gpu::LaunchKernel(
      fp_sums_kernel<TensorDataType, block_size>,
      grid_dims, block_dims, 0, multisync,
      local_mini_batch_size, num_channels, channel_size,
      local_input.LockedBuffer(), local_input.LDim(),
      local_sums.Buffer(), local_sums.LDim(),
      local_sqsums.Buffer(), local_sqsums.LDim());
  }

  // Normalize output
  if (!local_output.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_output),
                                       gpu::get_sync_info(local_workspace),
                                       gpu::get_sync_info(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_mini_batch_size;
    hydrogen::gpu::LaunchKernel(
      fp_output_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_mini_batch_size, num_channels, channel_size, epsilon,
      local_input.LockedBuffer(), local_input.LDim(),
      local_output.Buffer(), local_output.LDim(),
      local_sums.LockedBuffer(), local_sums.LDim(),
      local_sqsums.LockedBuffer(), local_sqsums.LDim());
  }

}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void instance_norm_layer<TensorDataType, Layout, Device>::fp_compute()
{
  const size_t num_channels = this->get_output_dims().front();
  const size_t channel_size = this->get_output_size() / num_channels;
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

namespace
{

/** Compute gradients w.r.t. per-channel statistics.
 *
 *  dL/dmean = - sum(dL/dy_i) / sqrt(var+epsilon)
 *
 *  dL/dvar = - sum(dL/dy_i * (x_i-mean)) * (var+epsilon)^(-3/2) / 2
 *
 *  On input, means_grad and vars_grad are filled with zeros.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (channel_size / bsize) x num_channels x mini_batch_size
 */
template <typename TensorDataType, size_t bdimx>
__global__ void bp_statistics_grad_kernel(
  size_t mini_batch_size,
  size_t num_channels,
  size_t channel_size,
  TensorDataType epsilon,
  const TensorDataType* __restrict__ input,
  size_t input_ldim,
  const TensorDataType* __restrict__ output_grad,
  size_t output_grad_ldim,
  const TensorDataType* sums,
  size_t sums_ldim,
  const TensorDataType* sqsums,
  size_t sqsums_ldim,
  TensorDataType* means_grad,
  size_t means_grad_ldim,
  TensorDataType* vars_grad,
  size_t vars_grad_ldim)
{

  // Indices and dimensions
  constexpr size_t bdimy = 1;
  constexpr size_t bdimz = 1;
  const size_t tid = threadIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;

  const TensorDataType mean_scale = 1. / channel_size;
  const TensorDataType var_correction = double(channel_size) / (channel_size - 1);
  for (size_t k = gidz; k < mini_batch_size; k += nthreadsz) {
    for (size_t j = gidy; j < num_channels; j += nthreadsy) {

      // Compute statistics from sums
      const auto& sum = sums[j+k*sums_ldim];
      const auto& sqsum = sqsums[j+k*sqsums_ldim];
      const auto& mean = sum * mean_scale;
      const auto& sqmean = sqsum * mean_scale;
      auto var = (sqmean - mean*mean) * var_correction;
      var = gpu_lib::max(var, TensorDataType{0.});
      const auto& inv_stdev = gpu_lib::rsqrt(var + epsilon);

      // Accumulate sums and perform block-wide reduction
      using pair_t = thrust::pair<TensorDataType,TensorDataType>;
      using pair_sum_t = pair_sum<pair_t>;
      pair_t dmean_dvar(0,0);
      for (size_t i = gidx; i < channel_size; i += nthreadsx) {
        const auto& x = input[i + j*channel_size + k*input_ldim];
        const auto& dy = output_grad[i + j*channel_size + k*output_grad_ldim];
        dmean_dvar.first += dy;
        dmean_dvar.second += dy * (x - mean);
      }
      dmean_dvar = gpu_lib::block_reduce<bdimx,bdimy,bdimz,pair_t,pair_sum_t>(dmean_dvar);

      // Output result to global memory
      if (tid == 0) {
        const TensorDataType dmean = -dmean_dvar.first * inv_stdev;
        const TensorDataType dvar = -dmean_dvar.second * inv_stdev*inv_stdev*inv_stdev / 2;
        gpu_lib::atomic_add(&means_grad[j+k*means_grad_ldim], dmean);
        gpu_lib::atomic_add(&vars_grad[j+k*vars_grad_ldim], dvar);
      }

    }
  }

}

/** Compute gradients w.r.t. input.
 *
 *  dL/dx_i = ( dL/dy_i / sqrt(var+epsilon)
 *              + dL/dmean / n
 *              + dL/dvar * (x_i - mean) * 2/(n-1) )
 *
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (channel_size / bdimx) x (num_channels / bdimy) x (mini_batch_size / bdimz)
 */
template <typename TensorDataType>
__global__ void bp_input_grad_kernel(
  size_t mini_batch_size,
  size_t num_channels,
  size_t channel_size,
  TensorDataType epsilon,
  const TensorDataType* __restrict__ input,
  size_t input_ldim,
  const TensorDataType* __restrict__ output_grad,
  size_t output_grad_ldim,
  TensorDataType* __restrict__ input_grad,
  size_t input_grad_ldim,
  const TensorDataType* __restrict__ sums,
  size_t sums_ldim,
  const TensorDataType* __restrict__ sqsums,
  size_t sqsums_ldim,
  const TensorDataType* means_grad,
  size_t means_grad_ldim,
  const TensorDataType* vars_grad,
  size_t vars_grad_ldim)
{

  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;

  const TensorDataType mean_scale = 1. / channel_size;
  const TensorDataType var_correction = double(channel_size) / (channel_size - 1);
  for (size_t k = gidz; k < mini_batch_size; k += nthreadsz) {
    for (size_t j = gidy; j < num_channels; j += nthreadsy) {
      const auto& sum = sums[j+k*sums_ldim];
      const auto& sqsum = sqsums[j+k*sqsums_ldim];
      const auto& mean = sum * mean_scale;
      const auto& sqmean = sqsum * mean_scale;
      auto var = (sqmean - mean*mean) * var_correction;
      var = gpu_lib::max(var, TensorDataType{0.});
      const auto& inv_stdev = gpu_lib::rsqrt(var + epsilon);
      const auto& dmean = means_grad[j+k*means_grad_ldim];
      const auto& dvar = vars_grad[j+k*vars_grad_ldim];
      for (size_t i = gidx; i < channel_size; i += nthreadsx) {
        const auto& x = input[i + j*channel_size + k*input_ldim];
        const auto& dy = output_grad[i + j*channel_size + k*output_grad_ldim];
        auto& dx = input_grad[i + j*channel_size + k*input_grad_ldim];
        dx = (dy * inv_stdev
              + dmean * mean_scale
              + dvar * (x - mean) * 2 * mean_scale * var_correction);
      }
    }
  }

}

/** @brief Backprop */
template <typename TensorDataType>
void bp_impl(lbann_comm& comm,
             size_t num_channels,
             size_t channel_size,
             TensorDataType epsilon,
             const El::AbstractDistMatrix<TensorDataType>& input,
             const El::AbstractDistMatrix<TensorDataType>& output_grad,
             El::AbstractDistMatrix<TensorDataType>& input_grad,
             const El::Matrix<TensorDataType, El::Device::GPU>& local_workspace)
{

  // Local matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(output_grad.LockedMatrix());
  auto& local_input_grad = dynamic_cast<LocalMat&>(input_grad.Matrix());
  const auto local_sums = El::LockedView(local_workspace,
                                         El::IR(0, num_channels),
                                         El::ALL);
  const auto local_sqsums = El::LockedView(local_workspace,
                                           El::IR(num_channels, 2*num_channels),
                                           El::ALL);

  // Dimensions
  const size_t local_mini_batch_size = local_input.Width();

  // Trivial case if channel size is 1
  // Note: Output is constant, so error signal is zero.
  if (channel_size <= 1) {
    El::Zero(input_grad);
    return;
  }

  // Compute gradient w.r.t. statistics
  LocalMat local_statistics_grad;
  El::Zeros(local_statistics_grad, 2*num_channels, local_mini_batch_size);
  auto local_means_grad = El::View(local_statistics_grad,
                                   El::IR(0, num_channels),
                                   El::ALL);
  auto local_vars_grad = El::View(local_statistics_grad,
                                  El::IR(num_channels, 2*num_channels),
                                  El::ALL);
  if (!local_output_grad.IsEmpty()) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_statistics_grad),
                        gpu::get_sync_info(local_workspace),
                        gpu::get_sync_info(local_output_grad),
                        gpu::get_sync_info(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_mini_batch_size;
    hydrogen::gpu::LaunchKernel(
      bp_statistics_grad_kernel<TensorDataType, block_size>,
      grid_dims, block_dims, 0, multisync,
      local_mini_batch_size, num_channels, channel_size, epsilon,
      local_input.LockedBuffer(), local_input.LDim(),
      local_output_grad.LockedBuffer(), local_output_grad.LDim(),
      local_sums.LockedBuffer(), local_sums.LDim(),
      local_sqsums.LockedBuffer(), local_sqsums.LDim(),
      local_means_grad.Buffer(), local_means_grad.LDim(),
      local_vars_grad.Buffer(), local_vars_grad.LDim());
  }

  // Compute gradient w.r.t. input
  if (!local_input_grad.IsEmpty()) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_input_grad),
                        gpu::get_sync_info(local_workspace),
                        gpu::get_sync_info(local_statistics_grad),
                        gpu::get_sync_info(local_output_grad),
                        gpu::get_sync_info(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_mini_batch_size;
    hydrogen::gpu::LaunchKernel(
      bp_input_grad_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_mini_batch_size, num_channels, channel_size, epsilon,
      local_input.LockedBuffer(), local_input.LDim(),
      local_output_grad.LockedBuffer(), local_output_grad.LDim(),
      local_input_grad.Buffer(), local_input_grad.LDim(),
      local_sums.LockedBuffer(), local_sums.LDim(),
      local_sqsums.LockedBuffer(), local_sqsums.LDim(),
      local_means_grad.LockedBuffer(), local_means_grad.LDim(),
      local_vars_grad.LockedBuffer(), local_vars_grad.LDim());
  }

}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void instance_norm_layer<TensorDataType, Layout, Device>::bp_compute()
{
  const size_t num_channels = this->get_output_dims().front();
  const size_t channel_size = this->get_output_size() / num_channels;
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
// Explicit template instantiation
// =============================================

#define PROTO(T)                                        \
  template class instance_norm_layer<                   \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
