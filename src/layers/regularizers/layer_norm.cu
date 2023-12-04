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
#include "lbann/comm_impl.hpp"
#include "lbann/layers/regularizers/layer_norm.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

#include <thrust/pair.h>

namespace lbann {

namespace {

/** Functor for adding @c thrust::pair objects. */
template <typename Pair>
struct pair_sum
{
  __device__ __forceinline__ Pair operator()(const Pair& x, const Pair& y)
  {
    return Pair(x.first + y.first, x.second + y.second);
  }
};

/** Accumulate sums and sums of squares for each data sample.
 *
 *  On input, sums and sqsums are filled with zeros.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (local_sample_size / bsize) x local_num_samples x 1
 */
template <size_t bdimx, typename TensorDataType>
__global__ void fp_sums_kernel(size_t local_num_samples,
                               size_t normalization_size,
                               size_t num_normalized,
                               size_t normalization_stride,
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
  const size_t tid = threadIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;

  for (size_t k = gidz; k < local_num_samples; k += nthreadsz) {
    for (size_t i = gidy; i < num_normalized; i += nthreadsy) {

      // Accumulate sums and perform block-wide reduction
      using pair_t = thrust::pair<TensorDataType, TensorDataType>;
      using pair_sum_t = pair_sum<pair_t>;
      pair_t sum_sqsum(0, 0);
      for (size_t j = gidx; j < normalization_size; j += nthreadsx) {
        const auto& x = vals[k * vals_ldim + i * normalization_stride + j];
        sum_sqsum.first += x;
        sum_sqsum.second += x * x;
      }
      sum_sqsum =
        gpu_lib::block_reduce<bdimx, bdimy, bdimz, pair_t, pair_sum_t>(
          sum_sqsum);

      // Output result to global memory
      if (tid == 0) {
        gpu_lib::atomic_add(&sums[i + k * sums_stride], sum_sqsum.first);
        gpu_lib::atomic_add(&sqsums[i + k * sqsums_stride], sum_sqsum.second);
      }
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
__global__ void fp_statistics_kernel(size_t local_num_samples,
                                     size_t global_normalization_size,
                                     size_t num_normalized,
                                     TensorDataType* means,
                                     size_t means_stride,
                                     TensorDataType* vars,
                                     size_t vars_stride)
{

  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t i = gidy; i < local_num_samples; i += nthreadsy) {
    for (size_t j = gidx; j < num_normalized; j += nthreadsx) {
      const auto sum = means[i * means_stride + j];
      const auto sqsum = vars[i * vars_stride + j];
      const TensorDataType sample_size_dt =
        TensorDataType(global_normalization_size);
      const auto& mean = sum / sample_size_dt;
      const auto& sqmean = sqsum / sample_size_dt;
      const auto& var = (sqmean - mean * mean);
      means[i * means_stride + j] = mean;
      vars[i * vars_stride + j] = gpu_lib::max(var, TensorDataType(0.0));
    }
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
template <typename TensorDataType, bool HAS_SCALE, bool HAS_BIAS>
__global__ void fp_output_kernel(size_t local_num_samples,
                                 size_t normalization_size,
                                 size_t num_normalized,
                                 size_t normalization_stride,
                                 TensorDataType epsilon,
                                 const TensorDataType* __restrict__ input,
                                 size_t input_ldim,
                                 TensorDataType* __restrict__ output,
                                 size_t output_ldim,
                                 const TensorDataType* __restrict__ means,
                                 size_t means_stride,
                                 const TensorDataType* __restrict__ vars,
                                 size_t vars_stride,
                                 const TensorDataType* __restrict__ scale,
                                 const TensorDataType* __restrict__ bias)
{

  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;
  for (size_t i = gidz; i < local_num_samples; i += nthreadsz) {
    for (size_t j = gidy; j < num_normalized; j += nthreadsy) {
      const auto& mean = means[i * means_stride + j];
      const auto& var = vars[i * vars_stride + j];
      const auto& inv_stdev = gpu_lib::rsqrt(var + epsilon);
      for (size_t k = gidx; k < normalization_size; k += nthreadsx) {
        const auto& x = input[i * input_ldim + j * normalization_stride + k];
        auto& y = output[i * input_ldim + j * normalization_stride + k];
        auto result = (x - mean) * inv_stdev;
        if constexpr (HAS_SCALE)
          result *= scale[k];
        if constexpr (HAS_BIAS)
          result += bias[k];
        y = result;
      }
    }
  }
}

/** @brief Forward prop */
template <typename TensorDataType>
void fp_impl(lbann_comm& comm,
             TensorDataType epsilon,
             El::Int normalization_size,
             El::Int global_normalization_size,
             El::Int num_normalized,
             El::Int normalization_stride,
             const El::AbstractDistMatrix<TensorDataType>& input,
             El::AbstractDistMatrix<TensorDataType>& output,
             El::AbstractDistMatrix<TensorDataType>& statistics,
             const TensorDataType* local_scale,
             const TensorDataType* local_bias)
{
  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;

  // Workspace buffer
  statistics.Empty(false);
  statistics.AlignWith(input);
  statistics.Resize(2 * num_normalized, input.Width());

  // Local matrices
  const auto& local_input =
    dynamic_cast<const GPUMatType&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<GPUMatType&>(output.Matrix());
  auto& local_statistics = dynamic_cast<GPUMatType&>(statistics.Matrix());
  auto local_means =
    El::View(local_statistics, El::IR(0, num_normalized), El::ALL);
  auto local_vars = El::View(local_statistics,
                             El::IR(num_normalized, 2 * num_normalized),
                             El::ALL);

  // Dimensions
  const size_t local_num_samples = local_input.Width();
  const size_t local_sample_size = local_input.Height();

  // Trivial cases
  if (local_num_samples < 1) {
    return;
  }

  // Compute sums
  El::Zero(statistics);
  if (!local_input.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_statistics),
                                       gpu::get_sync_info(local_input));
    auto kernel = (normalization_size >= 256
                     ? fp_sums_kernel<256, TensorDataType>
                     : (normalization_size >= 128
                          ? fp_sums_kernel<128, TensorDataType>
                          : (normalization_size >= 64
                               ? fp_sums_kernel<64, TensorDataType>
                               : fp_sums_kernel<32, TensorDataType>)));
    size_t block_size =
      (normalization_size >= 256
         ? 256
         : (normalization_size >= 128 ? 128
                                      : (normalization_size >= 64 ? 64 : 32)));
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (normalization_size + block_size - 1) / block_size;
    grid_dims.y = num_normalized;
    grid_dims.z = local_num_samples;
    hydrogen::gpu::LaunchKernel(kernel,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                local_num_samples,
                                normalization_size,
                                num_normalized,
                                normalization_stride,
                                local_input.LockedBuffer(),
                                local_input.LDim(),
                                local_means.Buffer(),
                                local_means.LDim(),
                                local_vars.Buffer(),
                                local_vars.LDim());
  }
  comm.allreduce(statistics, statistics.RedundantComm(), El::mpi::SUM);

  // Compute statistics from sums
  if (global_normalization_size <= 1) {
    // local_means already has correct values
    El::Fill(local_vars, El::TypeTraits<TensorDataType>::One());
  }
  else if (!local_statistics.IsEmpty()) {
    auto sync_info = gpu::get_sync_info(local_statistics);
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (num_normalized + block_size - 1) / block_size;
    grid_dims.y = local_num_samples;
    hydrogen::gpu::LaunchKernel(fp_statistics_kernel<TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                sync_info,
                                local_num_samples,
                                global_normalization_size,
                                num_normalized,
                                local_means.Buffer(),
                                local_means.LDim(),
                                local_vars.Buffer(),
                                local_vars.LDim());
  }

  // Apply layer norm
  if (!local_output.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_output),
                                       gpu::get_sync_info(local_statistics),
                                       gpu::get_sync_info(local_input));
    El::Int block_size = min(El::Int(256), normalization_size);
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    hydrogen::gpu::LaunchKernel(layer_norm_fp_output_kernel<TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                local_num_samples,
                                local_sample_size,
                                epsilon,
                                local_input.LockedBuffer(),
                                local_input.LDim(),
                                local_output.Buffer(),
                                local_output.LDim(),
                                local_means.LockedBuffer(),
                                local_means.LDim(),
                                local_vars.LockedBuffer(),
                                local_vars.LDim());
  }
}


/** @brief Backprop */
template <typename TensorDataType>
void bp_impl(lbann_comm& comm,
             TensorDataType epsilon,
             El::Int normalization_size,
             El::Int global_normalization_size,
             El::Int num_normalized,
             El::Int normalization_stride,
             const El::AbstractDistMatrix<TensorDataType>& input,
             const El::AbstractDistMatrix<TensorDataType>& output_grad,
             El::AbstractDistMatrix<TensorDataType>& input_grad,
             const El::AbstractDistMatrix<TensorDataType>& statistics,
             El::AbstractDistMatrix<TensorDataType>& statistics_grad,
             const TensorDataType* local_scale,
             TensorDataType* scale_grad,
             TensorDataType* bias_grad)
{
  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;

  // Workspace buffer
  statistics_grad.Empty(false);
  statistics_grad.AlignWith(input);
  statistics_grad.Resize(2 * num_normalized, input.Width());

  // Local matrices
  const auto& local_input =
    dynamic_cast<const GPUMatType&>(input.LockedMatrix());
  const auto& local_output_grad =
    dynamic_cast<const GPUMatType&>(output_grad.LockedMatrix());
  auto& local_input_grad = dynamic_cast<GPUMatType&>(input_grad.Matrix());
  const auto& local_statistics =
    dynamic_cast<const GPUMatType&>(statistics.LockedMatrix());
  const auto local_means =
    El::LockedView(local_statistics, El::IR(0, num_normalized), El::ALL);
  const auto local_vars =
    El::LockedView(local_statistics,
                   El::IR(num_normalized, 2 * num_normalized),
                   El::ALL);
  auto& local_statistics_grad =
    dynamic_cast<GPUMatType&>(statistics_grad.Matrix());
  auto local_means_grad =
    El::View(local_statistics_grad, El::IR(0, num_normalized), El::ALL);
  auto local_vars_grad = El::View(local_statistics_grad,
                                  El::IR(num_normalized, 2 * num_normalized),
                                  El::ALL);

  // Dimensions
  const size_t local_num_samples = local_input.Width();

  // Trivial case if sample size <= 1
  // Note: Output is constant, so error signal is zero.
  if (global_normalization_size <= 1) {
    El::Zero(input_grad);
    return;
  }

  // Compute gradient w.r.t. statistics
  El::Zero(statistics_grad);
  if (!local_output_grad.IsEmpty()) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_statistics_grad),
                        gpu::get_sync_info(local_output_grad),
                        gpu::get_sync_info(local_statistics),
                        gpu::get_sync_info(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_sample_size + block_size - 1) / block_size;
    grid_dims.y = local_num_samples;

    auto kernel =
      ((!scale_grad && !bias_grad)
         ? bp_statistics_grad_kernel<block_size, TensorDataType, false, false>
         : ((scale_grad && !bias_grad)
              ? bp_statistics_grad_kernel<block_size,
                                          TensorDataType,
                                          true,
                                          false>
              : ((!scale_grad && bias_grad)
                   ? bp_statistics_grad_kernel<block_size,
                                               TensorDataType,
                                               false,
                                               true>
                   : bp_statistics_grad_kernel<block_size,
                                               TensorDataType,
                                               true,
                                               true>)));
    hydrogen::gpu::LaunchKernel(kernel,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                local_num_samples,
                                normalization_size,
                                num_normalized,
                                normalization_stride,
                                epsilon,
                                local_input.LockedBuffer(),
                                local_input.LDim(),
                                local_output_grad.LockedBuffer(),
                                local_output_grad.LDim(),
                                local_means.LockedBuffer(),
                                local_means.LDim(),
                                local_vars.LockedBuffer(),
                                local_vars.LDim(),
                                local_means_grad.Buffer(),
                                local_means_grad.LDim(),
                                local_vars_grad.Buffer(),
                                local_vars_grad.LDim(),
                                local_scale,
                                scale_grad,
                                bias_grad);

    hydrogen::gpu::LaunchKernel(
      layer_norm_bp_statistics_grad_kernel<block_size, TensorDataType>,
      grid_dims,
      block_dims,
      0,
      multisync,
      local_num_samples,
      local_sample_size,
      epsilon,
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_output_grad.LockedBuffer(),
      local_output_grad.LDim(),
      local_means.LockedBuffer(),
      local_means.LDim(),
      local_vars.LockedBuffer(),
      local_vars.LDim(),
      local_means_grad.Buffer(),
      local_means_grad.LDim(),
      local_vars_grad.Buffer(),
      local_vars_grad.LDim());
  }
  comm.allreduce(statistics_grad,
                 statistics_grad.RedundantComm(),
                 El::mpi::SUM);

  // Compute gradient w.r.t. input
  if (!local_input_grad.IsEmpty()) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_statistics_grad),
                        gpu::get_sync_info(local_output_grad),
                        gpu::get_sync_info(local_statistics),
                        gpu::get_sync_info(local_input));
    El::Int block_size = min(El::Int(256), normalization_size);
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_sample_size + block_size - 1) / block_size;
    grid_dims.y = local_num_samples;

    auto kernel = (local_scale ? bp_input_grad_kernel<TensorDataType, true>
                               : bp_input_grad_kernel<TensorDataType, false>);
    hydrogen::gpu::LaunchKernel(kernel,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                local_num_samples,
                                global_normalization_size,
                                normalization_size,
                                num_normalized,
                                normalization_stride,
                                epsilon,
                                local_input.LockedBuffer(),
                                local_input.LDim(),
                                local_output_grad.LockedBuffer(),
                                local_output_grad.LDim(),
                                local_input_grad.Buffer(),
                                local_input_grad.LDim(),
                                local_means.LockedBuffer(),
                                local_means.LDim(),
                                local_vars.LockedBuffer(),
                                local_vars.LDim(),
                                local_means_grad.LockedBuffer(),
                                local_means_grad.LDim(),
                                local_vars_grad.LockedBuffer(),
                                local_vars_grad.LDim(),
                                local_scale);
  }
}

} // namespace

// =========================================================
// DistConv-Adapter member implementation
// =========================================================

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_distconv_adapter<TensorDataType, Layout, Device> fp_compute()
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_distconv_adapter<TensorDataType, Layout, Device> bp_compute()
{}
#endif // LBANN_HAS_DISTCONV

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
  El::Int norm_size, global_norm_size, num_norm, norm_stride;
  this->get_normdims(norm_size, global_norm_size, num_norm, norm_stride);

#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    this->get_distconv_adapter().fp_compute();
    return;
  }
#endif // LBANN_HAS_DISTCONV 
  fp_impl(*this->get_comm(),
          this->m_epsilon,
          norm_size,
          global_norm_size,
          num_norm,
          norm_stride,
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
  TensorDataType* scale_grad = nullptr;
  TensorDataType* bias_grad = nullptr;

  if (m_scale) {
    scale_weights = this->weights_values(0).LockedMatrix().LockedBuffer();
    El::Zero(*this->m_scale_gradient);
    scale_grad = this->m_scale_gradient->Buffer();
  }

  if (m_bias) {
    El::Zero(*this->m_bias_gradient);
    bias_grad = this->m_bias_gradient->Buffer();
  }

  El::Int norm_size, global_norm_size, num_norm, norm_stride;
  this->get_normdims(norm_size, global_norm_size, num_norm, norm_stride);

  // Compute backpropagation

void layer_norm_layer<TensorDataType, Layout, Device>::bp_compute()
{
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    this->get_distconv_adapter().bp_compute();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  bp_impl(*this->get_comm(),
          this->m_epsilon,
          norm_size,
          global_norm_size,
          num_norm,
          norm_stride,
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
                                  El::Device::GPU>;                            \
  template class layer_norm_layer<T,                                           \
                                  data_layout::MODEL_PARALLEL,                 \
                                  El::Device::GPU>

#include "lbann/macros/instantiate.hpp"

} // namespace lbann
