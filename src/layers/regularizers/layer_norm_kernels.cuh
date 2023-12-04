////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYERS_REGULARIZERS_NORM_LAYER_KERNELS
#define LBANN_LAYERS_REGULARIZERS_NORM_LAYER_KERNELS
#include "lbann/comm_impl.hpp"
#include "lbann/layers/regularizers/layer_norm.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include <thrust/pair.h>

namespace lbann {

/** Functor for adding @c thrust::pair objects. */
template <typename Pair>
struct pair_sum
{
  __device__ __forceinline__ Pair operator()(const Pair& x, const Pair& y)
  {
    return Pair(x.first + y.first, x.second + y.second);
  }
};

// =========================================================
// Forward prop
// =========================================================

/** Accumulate sums and sums of squares for each data sample.
 *
 *  On input, sums and sqsums are filled with zeros.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (local_sample_size / bsize) x local_num_samples x 1
 */
template <size_t bdimx, typename TensorDataType>
__global__ void
layer_norm_fp_sums_kernel(size_t local_num_samples,
                          size_t local_sample_size,
                          const TensorDataType* __restrict__ vals,
                          size_t vals_ldim,
                          TensorDataType* sums,
                          size_t sums_stride,
                          TensorDataType* sqsums,
                          size_t sqsums_stride)
{

  // Indices and dimensions
  constexpr size_t bdimy = 1;
  constexpr size_t bdimz = 1;
  const size_t tid = threadIdx.x + blockDim.x * threadIdx.y;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;

  for (size_t i = gidy; i < local_num_samples; i += nthreadsy) {

    // Accumulate sums and perform block-wide reduction
    using pair_t = thrust::pair<TensorDataType, TensorDataType>;
    using pair_sum_t = pair_sum<pair_t>;
    pair_t sum_sqsum(0, 0);
    for (size_t j = gidx; j < local_sample_size; j += nthreadsx) {
      const auto& x = vals[i * vals_ldim + j];
      sum_sqsum.first += x;
      sum_sqsum.second += x * x;
    }
    sum_sqsum =
      gpu_lib::block_reduce<bdimx, bdimy, bdimz, pair_t, pair_sum_t>(sum_sqsum);

    // Output result to global memory
    if (tid == 0) {
      gpu_lib::atomic_add(&sums[i * sums_stride], sum_sqsum.first);
      gpu_lib::atomic_add(&sqsums[i * sqsums_stride], sum_sqsum.second);
    }
  }
}

/** Compute per-sample statistics.
 *
 *  mean = sum(x_i) / n
 *
 *  var = ( sum(x_i^2)/n - mean^2 )
 *
 *  On input, means contains per-sample sums and vars contains
 *  per-sample sums of squares.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (local_num_samples / bsize) x 1 x 1
 */
template <typename TensorDataType>
__global__ void layer_norm_fp_statistics_kernel(unsigned long long sample_size,
                                                size_t local_num_samples,
                                                TensorDataType* means,
                                                size_t means_stride,
                                                TensorDataType* vars,
                                                size_t vars_stride)
{

  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  for (size_t i = gid; i < local_num_samples; i += nthreads) {
    const auto sum = means[i * means_stride];
    const auto sqsum = vars[i * means_stride];
    const TensorDataType sample_size_dt = TensorDataType(sample_size);
    const auto& mean = sum / sample_size_dt;
    const auto& sqmean = sqsum / sample_size_dt;
    const auto& var = (sqmean - mean * mean);
    means[i * means_stride] = mean;
    vars[i * vars_stride] = gpu_lib::max(var, TensorDataType(0.0));
  }
}

/** Compute outputs.
 *
 *  y_i = (x_i - mean) / sqrt(var + epsilon)
 *
 *  Block dimensions: bdimx x bdimy x 1
 *
 *  Grid dimensions: (local_sample_size / bdimx) x (local_num_samples / bdimy) x
 * 1
 */
template <typename TensorDataType>
__global__ void
layer_norm_fp_output_kernel(size_t local_num_samples,
                            size_t local_sample_size,
                            TensorDataType epsilon,
                            const TensorDataType* __restrict__ input,
                            size_t input_ldim,
                            TensorDataType* __restrict__ output,
                            size_t output_ldim,
                            const TensorDataType* means,
                            size_t means_stride,
                            const TensorDataType* vars,
                            size_t vars_stride)
{

  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t i = gidy; i < local_num_samples; i += nthreadsy) {
    const auto& mean = means[i * means_stride];
    const auto& var = vars[i * vars_stride];
    const auto& inv_stdev = gpu_lib::rsqrt(var + epsilon);
    for (size_t j = gidx; j < local_sample_size; j += nthreadsx) {
      const auto& x = input[i * input_ldim + j];
      auto& y = output[i * output_ldim + j];
      y = (x - mean) * inv_stdev;
    }
  }
}

/** Compute gradients w.r.t. per-sample statistics.
 *
 *  dL/dmean = - sum(dL/dy_i) / sqrt(var+epsilon)
 *
 *  dL/dvar = - sum(dL/dy_i * (x_i-mean)) * (var+epsilon)^(-3/2) / 2
 *
 *  On input, means_grad and vars_grad are filled with zeros.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (local_sample_size / bsize) x local_num_samples x 1
 */
template <size_t bdimx, typename TensorDataType>
__global__ void layer_norm_bp_statistics_grad_kernel(
  size_t local_num_samples,
  size_t local_sample_size,
  TensorDataType epsilon,
  const TensorDataType* __restrict__ input,
  size_t input_ldim,
  const TensorDataType* __restrict__ output_grad,
  size_t output_grad_ldim,
  const TensorDataType* means,
  size_t means_stride,
  const TensorDataType* vars,
  size_t vars_stride,
  TensorDataType* means_grad,
  size_t means_grad_stride,
  TensorDataType* vars_grad,
  size_t vars_grad_stride)
{

  // Indices and dimensions
  constexpr size_t bdimy = 1;
  constexpr size_t bdimz = 1;
  const size_t tid = threadIdx.x + blockDim.x * threadIdx.y;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;

  for (size_t i = gidy; i < local_num_samples; i += nthreadsy) {

    // Accumulate sums and perform block-wide reduction
    using pair_t = thrust::pair<TensorDataType, TensorDataType>;
    using pair_sum_t = pair_sum<pair_t>;
    pair_t sums(0, 0);
    const auto& mean = means[i * means_stride];
    for (size_t j = gidx; j < local_sample_size; j += nthreadsx) {
      const auto& x = input[i * input_ldim + j];
      const auto& dy = output_grad[i * output_grad_ldim + j];
      sums.first += dy;
      sums.second += dy * (x - mean);
    }
    sums = gpu_lib::block_reduce<bdimx, bdimy, bdimz, pair_t, pair_sum_t>(sums);

    // Output result to global memory
    if (tid == 0) {
      const auto& var = vars[i * vars_stride];
      const auto& inv_stdev = gpu_lib::rsqrt(var + epsilon);
      const TensorDataType dmean = -sums.first * inv_stdev;
      const TensorDataType dvar =
        -sums.second * inv_stdev * inv_stdev * inv_stdev / TensorDataType(2);
      gpu_lib::atomic_add(&means_grad[i * means_grad_stride], dmean);
      gpu_lib::atomic_add(&vars_grad[i * vars_grad_stride], dvar);
    }
  }
}

/** Compute gradients w.r.t. input.
 *
 *  dL/dx_i = ( dL/dy_i / sqrt(var+epsilon)
 *              + dL/dmean / n
 *              + dL/dvar * (x_i - mean) * 2/(n-1) )
 *
 *  Block dimensions: bdimx x bdimy x 1
 *
 *  Grid dimensions: (local_sample_size / bdimx) x (local_num_samples / bdimy) x
 * 1
 */
template <typename TensorDataType>
__global__ void
layer_norm_bp_input_grad_kernel(unsigned long long sample_size,
                                size_t local_num_samples,
                                size_t local_sample_size,
                                TensorDataType epsilon,
                                const TensorDataType* __restrict__ input,
                                size_t input_ldim,
                                const TensorDataType* __restrict__ output_grad,
                                size_t output_grad_ldim,
                                TensorDataType* __restrict__ input_grad,
                                size_t input_grad_ldim,
                                const TensorDataType* __restrict__ means,
                                size_t means_stride,
                                const TensorDataType* __restrict__ vars,
                                size_t vars_stride,
                                const TensorDataType* means_grad,
                                size_t means_grad_stride,
                                const TensorDataType* vars_grad,
                                size_t vars_grad_stride)
{

  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t i = gidy; i < local_num_samples; i += nthreadsy) {
    const auto& mean = means[i * means_stride];
    const auto& var = vars[i * vars_stride];
    const auto& inv_stdev = gpu_lib::rsqrt(var + epsilon);
    const auto& dmean = means_grad[i * means_grad_stride];
    const auto& dvar = vars_grad[i * vars_grad_stride];
    for (size_t j = gidx; j < local_sample_size; j += nthreadsx) {
      const auto& x = input[i * input_ldim + j];
      const auto& dy = output_grad[i * output_grad_ldim + j];
      auto& dx = input_grad[i * input_grad_ldim + j];
      dx =
        (dy * inv_stdev + dmean / TensorDataType(sample_size) +
         dvar * (x - mean) * TensorDataType(2) / TensorDataType(sample_size));
    }
  }
}

} // namespace lbann

#endif // LBANN_LAYERS_REGULARIZERS_NORM_LAYER_KERNELS