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

#define LBANN_BATCH_NORMALIZATION_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/weights/weights_helpers.hpp"
#include "lbann/utils/cuda.hpp"

namespace lbann {

namespace {

/** Functor for adding arrays. */
template <typename T, size_t N>
struct array_sum
{
  using ArrayType = cuda::array<T,N>;
  __device__ __forceinline__
  ArrayType operator()(const ArrayType& x, const ArrayType& y)
  {
    ArrayType sum;
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      sum[i] = x[i] + y[i];
    }
    return sum;
  }
};

/** Accumulate sums and sums of squares for each channel.
 *
 *  On input, sums and sqsums are assumed to be filled with zeros.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (channel_size / bsize) x num_channels x 1
 */
template <typename TensorDataType, int bdimx>
__global__ void fp_sums_kernel(
  int mini_batch_size,
  int num_channels,
  int channel_size,
  const TensorDataType * __restrict__ data, int data_ldim,
        TensorDataType * __restrict__ sums,
        TensorDataType * __restrict__ sqsums) {

  // Indices and dimensions
  constexpr int bdimy = 1;
  constexpr int bdimz = 1;
  const auto& tid = threadIdx.x;
  const auto& gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& gidy = blockIdx.y;
  const auto& nthreadsx = blockDim.x * gridDim.x;
  const auto& nthreadsy = gridDim.y;

  for (int channel = gidy; channel < num_channels; channel += nthreadsy) {

    // Accumulate sums and perform block-wide reduction
    using array_t = cuda::array<TensorDataType,2>;
    using array_sum_t = array_sum<TensorDataType,2>;
    array_t sum_sqsum;
    sum_sqsum[0] = TensorDataType(0);
    sum_sqsum[1] = TensorDataType(0);
    for (int i = gidx; i < channel_size; i += nthreadsx) {
      for (int j = 0; j < mini_batch_size; ++j) {
        const auto& x = data[i + channel*channel_size + j*data_ldim];
        sum_sqsum[0] += x;
        sum_sqsum[1] += x * x;
      }
    }
    sum_sqsum = cuda::block_reduce<bdimx,bdimy,bdimz,array_t,array_sum_t>(sum_sqsum);

    // Output result to global memory
    if (tid == 0) {
      cuda::atomic_add(&sums[channel], sum_sqsum[0]);
      cuda::atomic_add(&sqsums[channel], sum_sqsum[1]);
    }

  }

}

/** Compute statistics for each channel.
 *
 *  On input, global_mean and global_var are assumed to contain sums
 *  and squares of sums, respectively.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (num_channels / bsize) x 1 x 1
 */
template <typename TensorDataType>
__global__ void fp_statistics_kernel(
  int num_sums,
  int num_per_sum,
  TensorDataType epsilon,
  TensorDataType decay,
  TensorDataType * __restrict__ global_mean,
  TensorDataType * __restrict__ global_var,
  TensorDataType * __restrict__ global_running_mean,
  TensorDataType * __restrict__ global_running_var) {

  const auto& gid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& num_threads = blockDim.x * gridDim.x;
  for (auto i = gid; i < num_sums; i += num_threads) {

    TensorDataType num_per_sum_dt = TensorDataType(num_per_sum);
    // Compute mean and variance
    const auto& mean = global_mean[i] / num_per_sum_dt;
    const auto& sqmean = global_var[i] / num_per_sum_dt;
    auto var = num_per_sum_dt * (sqmean - mean * mean) / TensorDataType(num_per_sum - 1);
    var = var > epsilon ? var : epsilon;
    global_mean[gid] = mean;
    global_var[gid] = var;

    // Compute running statistics
    auto& running_mean = global_running_mean[gid];
    auto& running_var = global_running_var[gid];
    running_mean = decay * running_mean + (TensorDataType(1.0) - decay) * mean;
    running_var = decay * running_var + (TensorDataType(1.0) - decay) * var;

  }

}

/** Compute outputs.
 *
 *  y_i = (x_i - mean) / sqrt(var + epsilon)
 *
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (channel_size / bdimx) x (mini_batch_size / bdimy) x (num_channels / bdimz)
 *
 */
template <typename TensorDataType>
__global__ void fp_output_kernel(
  int mini_batch_size,
  int num_channels,
  int channel_size,
  const TensorDataType * __restrict__ global_input, int input_ldim,
  const TensorDataType * __restrict__ global_mean,
  const TensorDataType * __restrict__ global_var,
  TensorDataType epsilon,
  const TensorDataType * __restrict__ global_scale,
  const TensorDataType * __restrict__ global_bias,
        TensorDataType * __restrict__ global_output, int output_ldim) {

  // Indices and dimensions
  const auto& gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const auto& gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const auto& nthreadsx = blockDim.x * gridDim.x;
  const auto& nthreadsy = blockDim.y * gridDim.y;
  const auto& nthreadsz = blockDim.z * gridDim.z;

  for (auto k = gidz; k < num_channels; k += nthreadsz) {
    const auto& mean = global_mean[k];
    const auto& var = global_var[k];
    const auto& inv_stdev = cuda::rsqrt(var + epsilon);
    const auto& scale = global_scale[k];
    const auto& bias = global_bias[k];
    for (auto j = gidy; j < mini_batch_size; j += nthreadsy) {
      for (auto i = gidx; i < channel_size; i += nthreadsx) {
        const auto& x = global_input[i + k*channel_size + j*input_ldim];
        const auto& xhat = (x - mean) * inv_stdev;
        const auto& y = scale * xhat + bias;
        global_output[i + k*channel_size + j*output_ldim] = y;
      }
    }
  }

}

/** Compute gradients w.r.t. statistics and affine transform.
 *
 *  dL/dscale = sum(dL/dy_i * xhat_i)
 *
 *  dL/dbias = sum(dL/dy_i)
 *
 *  dL/dmean = - sum(dL/dy_i) / sqrt(var+epsilon)
 *
 *  dL/dvar = - sum(dL/dy_i * (x_i-mean)) * (var+epsilon)^(-3/2) / 2
 *
 *  On input, means_grad and vars_grad are filled with zeros.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (channel_size / bsize) x num_channels x 1
 */
template <typename TensorDataType, int bdimx>
__global__ void bp_statistics_grad_kernel(
  int mini_batch_size,
  int num_channels,
  int channel_size,
  const TensorDataType * __restrict__ global_input,
  int input_ldim,
  const TensorDataType * __restrict__ global_gradient_wrt_output,
  int gradient_wrt_output_ldim,
  const TensorDataType * __restrict__ global_mean,
  const TensorDataType * __restrict__ global_var,
  TensorDataType epsilon,
  const TensorDataType * __restrict__ global_scale,
        TensorDataType * __restrict__ global_dscale,
        TensorDataType * __restrict__ global_dbias,
        TensorDataType * __restrict__ global_dmean,
        TensorDataType * __restrict__ global_dvar) {

  // Indices and dimensions
  constexpr int bdimy = 1;
  constexpr int bdimz = 1;
  const auto& tid = threadIdx.x;
  const auto& gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& gidy = blockIdx.y;
  const auto& nthreadsx = blockDim.x * gridDim.x;
  const auto& nthreadsy = gridDim.y;

  for (int channel = gidy; channel < num_channels; channel += nthreadsy) {

    // Copy batch normalization parameters to private memory
    const auto& mean = global_mean[channel];
    const auto& var = global_var[channel];
    const auto& scale = global_scale[channel];

    // Compute useful constants
    const auto& inv_stdev = cuda::rsqrt(var + epsilon);
    const auto& dvar_factor = inv_stdev * inv_stdev * inv_stdev * TensorDataType(0.5);

    // Accumulate sums and perform block-wide reduction
    using array_t = cuda::array<TensorDataType,4>;
    using array_sum_t = array_sum<TensorDataType,4>;
    array_t sums;
    sums[0] = TensorDataType(0);
    sums[1] = TensorDataType(0);
    sums[2] = TensorDataType(0);
    sums[3] = TensorDataType(0);
    for (int i = gidx; i < channel_size; i += nthreadsx) {
      for (int j = 0; j < mini_batch_size; ++j) {
        const auto& x = global_input[i + channel*channel_size + j*input_ldim];
        const auto& xhat = (x - mean) * inv_stdev;
        const auto& dy = global_gradient_wrt_output[i
                                                    + channel*channel_size
                                                    + j*gradient_wrt_output_ldim];
        sums[0] += dy * xhat;
        sums[1] += dy;
        const auto& dxhat = dy * scale;
        sums[2] -= dxhat * inv_stdev;
        sums[3] -= dxhat * (x - mean) * dvar_factor;
      }
    }
    sums = cuda::block_reduce<bdimx,bdimy,bdimz,array_t,array_sum_t>(sums);

    // Output result to global memory
    if (tid == 0) {
      cuda::atomic_add(&global_dscale[channel], sums[0]);
      cuda::atomic_add(&global_dbias[channel], sums[1]);
      cuda::atomic_add(&global_dmean[channel], sums[2]);
      cuda::atomic_add(&global_dvar[channel], sums[3]);
    }

  }

}

/** Compute gradients w.r.t. input.
 *
 *  dL/dx_i = ( dL/dxhat_i / sqrt(var+epsilon)
 *              + dL/dmean / n
 *              + dL/dvar * (x_i - mean) * 2/(n-1) )
 *
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (channel_size / bdimx) x (mini_batch_size / bdimy) x (num_channels / bdimz)
 */
template <typename TensorDataType>
__global__ void bp_input_grad_kernel(
  int mini_batch_size,
  int num_channels,
  int channel_size,
  int num_per_sum,
  const TensorDataType * __restrict__ global_input,
  int input_ldim,
  const TensorDataType * __restrict__ global_gradient_wrt_output,
  int gradient_wrt_output_ldim,
  const TensorDataType * __restrict__ global_mean,
  const TensorDataType * __restrict__ global_var,
  TensorDataType epsilon,
  const TensorDataType * __restrict__ global_scale,
  const TensorDataType * __restrict__ global_dmean,
  const TensorDataType * __restrict__ global_dvar,
        TensorDataType * __restrict__ global_gradient_wrt_input,
  int gradient_wrt_input_ldim) {

  // Indices and dimensions
  const auto& gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const auto& gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const auto& nthreadsx = blockDim.x * gridDim.x;
  const auto& nthreadsy = blockDim.y * gridDim.y;
  const auto& nthreadsz = blockDim.z * gridDim.z;

  for (auto k = gidz; k < num_channels; k += nthreadsz) {
    const auto& mean = global_mean[k];
    const auto& var = global_var[k];
    const auto& inv_stdev = cuda::rsqrt(var + epsilon);
    const auto& scale = global_scale[k];
    const auto& dmean = global_dmean[k];
    const auto& dvar = global_dvar[k];
    const auto& dmean_term = dmean / TensorDataType(num_per_sum);
    const auto& dvar_term = dvar * TensorDataType(2) / TensorDataType(num_per_sum - 1);
    for (auto j = gidy; j < mini_batch_size; j += nthreadsy) {
      for (auto i = gidx; i < channel_size; i += nthreadsx) {
        const auto& x = global_input[i + k*channel_size + j*input_ldim];
        const auto& dy = global_gradient_wrt_output[i + k*channel_size + j*gradient_wrt_output_ldim];
        const auto& dxhat = dy * scale;
        auto& dx = global_gradient_wrt_input[i + k*channel_size + j*gradient_wrt_input_ldim];
        dx = dxhat * inv_stdev + dmean_term + dvar_term * (x - mean);
      }
    }
  }

}

} // namespace

#ifdef LBANN_HAS_DISTCONV

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::fp_compute() {
  assert_always(Dev == El::Device::GPU);
  assert_always(T_layout == data_layout::DATA_PARALLEL);

  using ValuesGetter = weights_details::SafeWeightsAccessor<TensorDataType>;

  auto &l = dynamic_cast<batch_normalization_layer<
    TensorDataType, T_layout, Dev>&>(this->layer());

  const bool is_training =
      l.m_model->get_execution_context().get_execution_mode() == execution_mode::training;
  auto& local_running_mean =
    ValuesGetter::mutable_values(l.get_weights(2)).Matrix();
  auto& local_running_var =
    ValuesGetter::mutable_values(l.get_weights(3)).Matrix();

  assert0(dc::tensor::View(
      m_scale, l.weights_values(0).LockedMatrix().LockedBuffer()));
  assert0(dc::tensor::View(
      m_bias, l.weights_values(1).LockedMatrix().LockedBuffer()));
  assert0(dc::tensor::View(
      m_running_mean, local_running_mean.Buffer()));
  assert0(dc::tensor::View(
      m_running_var, local_running_var.Buffer()));

  m_bn->forward_stage1(this->get_prev_activations(), m_mean,
                       m_var, is_training);

  if (l.m_statistics_group_size == 0) {
    l.m_comm->allreduce(*l.m_mean_and_var, l.m_mean_and_var->RedundantComm(),
                        El::mpi::SUM);
  } else if (l.m_statistics_group_size == 1) {
    // Local aggregation
  } else {
    LBANN_ERROR("statics_group_size must be either 0 or 1 for now.");
  }

  m_bn->forward_stage2(this->get_prev_activations(),
                       m_mean, m_var, m_running_mean,
                       m_running_var, m_scale, m_bias,
                       this->get_activations(), is_training);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::bp_compute() {
  assert_always(Dev == El::Device::GPU);
  assert_always(T_layout == data_layout::DATA_PARALLEL);

  auto &l = dynamic_cast<batch_normalization_layer<
    TensorDataType, T_layout, Dev>&>(this->layer());

  // Check execution mode
  const bool is_training =
      l.m_model->get_execution_context().get_execution_mode() == execution_mode::training;
  assert_always(is_training);

  assert0(dc::tensor::View(
      m_scale, l.weights_values(0).LockedMatrix().LockedBuffer()));

  m_bn->backward_stage1(this->get_prev_activations(),
                        this->get_prev_error_signals(),
                        m_mean, m_var, m_scale,
                        m_scale_gradient, m_bias_gradient,
                        m_mean_gradient, m_var_gradient);

  // Verbatim copy from bp_compute_gpu
  // Accumulate gradients
  if (is_training) {
    if (l.m_statistics_group_size == 0) {
      l.m_comm->allreduce(*l.m_mean_and_var_gradient,
                          l.m_mean_and_var_gradient->RedundantComm(),
                          El::mpi::SUM);
    }
  } else {
    Zero(*l.m_mean_and_var_gradient);
  }

  auto* scale_optimizer = l.get_weights(0).get_optimizer();
  if (scale_optimizer != nullptr) {
    scale_optimizer->add_to_gradient(*l.m_scale_gradient, TensorDataType{1}, true);
  }
  auto* bias_optimizer = l.get_weights(1).get_optimizer();
  if (bias_optimizer != nullptr) {
    bias_optimizer->add_to_gradient(*l.m_bias_gradient, TensorDataType{1}, true);
  }

  m_bn->backward_stage2(this->get_prev_activations(), this->get_prev_error_signals(),
                        m_mean, m_var, m_scale, m_mean_gradient, m_var_gradient,
                        this->get_error_signals());
}

#endif // LBANN_HAS_DISTCONV

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_layer<TensorDataType, T_layout, Dev>::fp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    get_distconv_adapter().fp_compute();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  const bool is_training = this->m_model->get_execution_context().get_execution_mode() == execution_mode::training;

  // CUDA objects
  CHECK_CUDA(cudaSetDevice(hydrogen::gpu::DefaultDevice()));
  auto&& stream = hydrogen::cuda::GetDefaultStream();

  // Matrices
  const auto& input = this->get_prev_activations();
  const auto& local_input = input.LockedMatrix();
  auto& local_output = this->get_local_activations();

  // Matrix parameters
  const auto& width = input.Width();
  const auto& local_width = local_input.Width();
  const auto& output_dims = this->get_output_dims();
  const auto& num_channels = output_dims[0];
  const auto& channel_size = this->get_output_size() / num_channels;

  // Compute statistics
  if (is_training) {
    using ValuesGetter = weights_details::SafeWeightsAccessor<TensorDataType>;

    // Local matrices
    auto& local_mean = this->m_mean_v->Matrix();
    auto& local_var = this->m_var_v->Matrix();
    auto& local_running_mean =
      ValuesGetter::mutable_values(this->get_weights(2)).Matrix();
    auto& local_running_var =
      ValuesGetter::mutable_values(this->get_weights(3)).Matrix();

    // Compute sums and sums of squares
    El::Zero(local_mean);
    El::Zero(local_var);
    if (!local_input.IsEmpty()) {
      constexpr int block_size = 256;
      dim3 block_dims, grid_dims;
      block_dims.x = block_size;
      grid_dims.x = (channel_size + block_size - 1) / block_size;
      grid_dims.y = El::Min(num_channels, 65535);
      fp_sums_kernel<TensorDataType, block_size>
        <<<grid_dims, block_dims, 0, stream>>>(
          local_width,
          num_channels,
          channel_size,
          local_input.LockedBuffer(), local_input.LDim(),
          local_mean.Buffer(),
          local_var.Buffer());
    }
    int num_per_sum;
    if (this->m_statistics_group_size == 0) {
      // Global statistics aggregation; allreduce on fused buffer.
      this->m_comm->allreduce(*this->m_mean_and_var, this->m_mean_and_var->RedundantComm(),
                        El::mpi::SUM);
      num_per_sum = channel_size * width;
    } else if (this->m_statistics_group_size == 1) {
      // Local aggregation, no allreduce needed.
      num_per_sum = channel_size * local_width;
    } else {
      // Grouped batchnorm. Allreduce on fused buffer.
      this->m_comm->allreduce(*this->m_mean_and_var,
                        this->m_comm->get_packed_group_comm(this->m_statistics_group_size),
                        El::mpi::SUM);
      if (this->m_num_per_sum_cache.count(width) == 0) {
        num_per_sum = channel_size * local_width;
        num_per_sum = this->m_comm->allreduce(
          num_per_sum, this->m_comm->get_packed_group_comm(this->m_statistics_group_size));
        this->m_num_per_sum_cache[width] = num_per_sum;
      } else {
        num_per_sum = this->m_num_per_sum_cache[width];
      }
    }

    // Compute minibatch statistics
    if (num_per_sum <= 1) {
      El::Fill(local_var, TensorDataType(1.0));
    } else if (num_channels > 0) {
      constexpr size_t block_dim = 256;
      const size_t grid_dim = El::Min((num_channels + block_dim - 1) / block_dim,
                                      65535);
      fp_statistics_kernel<<<grid_dim, block_dim, 0, stream>>>(
          num_channels, num_per_sum, this->m_epsilon, this->m_decay,
          local_mean.Buffer(), local_var.Buffer(),
          local_running_mean.Buffer(), local_running_var.Buffer());
    }

  }

  // Apply batch normalization
  const auto& local_scale = this->weights_values(0).LockedMatrix();
  const auto& local_bias = this->weights_values(1).LockedMatrix();
  const auto& local_mean = (is_training ?
                            this->m_mean_v->LockedMatrix() :
                            this->weights_values(2).LockedMatrix());
  const auto& local_var = (is_training ?
                           this->m_var_v->LockedMatrix() :
                           this->weights_values(3).LockedMatrix());
  if (!local_input.IsEmpty()) {
    constexpr int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = El::Min(local_width, 65535);
    grid_dims.z = El::Min(num_channels, 65535);
    fp_output_kernel
      <<<grid_dims, block_dims, 0, stream>>>(
        local_width, num_channels, channel_size,
        local_input.LockedBuffer(), local_input.LDim(),
        local_mean.LockedBuffer(), local_var.LockedBuffer(), this->m_epsilon,
        local_scale.LockedBuffer(), local_bias.LockedBuffer(),
        local_output.Buffer(), local_output.LDim());
  }

}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_layer<TensorDataType, T_layout, Dev>::bp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    get_distconv_adapter().bp_compute();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  const bool is_training = this->m_model->get_execution_context().get_execution_mode() == execution_mode::training;

  // CUDA objects
  CHECK_CUDA(cudaSetDevice(hydrogen::gpu::DefaultDevice()));
  auto&& stream = hydrogen::cuda::GetDefaultStream();

  // Matrices
  const auto& local_scale = this->weights_values(0).LockedMatrix();
  const auto& local_mean = (is_training ?
                            this->m_mean_v->LockedMatrix() :
                            this->weights_values(2).LockedMatrix());
  const auto& local_var = (is_training ?
                           this->m_var_v->LockedMatrix() :
                           this->weights_values(3).LockedMatrix());
  const auto& input = this->get_prev_activations();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
  auto& local_gradient_wrt_input = this->get_local_error_signals();
  auto& local_mean_gradient = this->m_mean_gradient_v->Matrix();
  auto& local_var_gradient = this->m_var_gradient_v->Matrix();
  auto& local_scale_gradient = this->m_scale_gradient->Matrix();
  auto& local_bias_gradient = this->m_bias_gradient->Matrix();

  // Matrix parameters
  const auto& width = input.Width();
  const auto& local_width = local_input.Width();
  const auto& output_dims = this->get_output_dims();
  const auto& num_channels = output_dims[0];
  const auto& channel_size = this->get_output_size() / num_channels;

  // Compute local gradients
  // Compute gradients w.r.t. batch norm parameters
  El::Zero(local_scale_gradient);
  El::Zero(local_bias_gradient);
  El::Zero(local_mean_gradient);
  El::Zero(local_var_gradient);
  if (!local_input.IsEmpty()) {
    constexpr int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = El::Min(num_channels, 65535);
    bp_statistics_grad_kernel<TensorDataType,block_size>
      <<<grid_dims, block_dims, 0, stream>>>(
        local_width, num_channels, channel_size,
        local_input.LockedBuffer(), local_input.LDim(),
        local_gradient_wrt_output.LockedBuffer(), local_gradient_wrt_output.LDim(),
        local_mean.LockedBuffer(), local_var.LockedBuffer(), this->m_epsilon,
        local_scale.LockedBuffer(),
        local_scale_gradient.Buffer(), local_bias_gradient.Buffer(),
        local_mean_gradient.Buffer(), local_var_gradient.Buffer());
  }

  // Accumulate gradients
  if (is_training) {
    if (this->m_statistics_group_size == 0) {
      // Global aggregation; allreduce on fused buffer.
      this->m_comm->allreduce(*this->m_mean_and_var_gradient,
                        this->m_mean_and_var_gradient->RedundantComm(),
                        El::mpi::SUM);
    } else if (this->m_statistics_group_size > 1) {
      // Grouped batchnorm; allreduce on fused buffer.
      this->m_comm->allreduce(*this->m_mean_and_var_gradient,
                        this->m_comm->get_packed_group_comm(this->m_statistics_group_size),
                        El::mpi::SUM);
    }
  } else {
    // Zero fused buffer.
    El::Zero(*this->m_mean_and_var_gradient);
  }
  auto* scale_optimizer = this->get_weights(0).get_optimizer();
  if (scale_optimizer != nullptr) {
    scale_optimizer->add_to_gradient(*this->m_scale_gradient, TensorDataType(1.0), true);
  }
  auto* bias_optimizer = this->get_weights(1).get_optimizer();
  if (bias_optimizer != nullptr) {
    bias_optimizer->add_to_gradient(*this->m_bias_gradient, TensorDataType(1.0), true);
  }

  // Compute error signal
  int num_per_sum;
  if (this->m_statistics_group_size == 0) {
    // Global statistics aggregation.
    num_per_sum = channel_size * width;
  } else if (this->m_statistics_group_size == 1) {
    // Local aggregation.
    num_per_sum = channel_size * local_width;
  } else {
    // Grouped batchnorm.
    num_per_sum = this->m_num_per_sum_cache[width];  // This was computed in FP.
  }
  if (num_per_sum <= 1) {
    El::Zero(local_gradient_wrt_input);
  } else if (!local_input.IsEmpty()) {
    constexpr int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = El::Min(local_width, 65535);
    grid_dims.z = El::Min(num_channels, 65535);
    bp_input_grad_kernel
      <<<grid_dims, block_dims, 0, stream>>>(
        local_width, num_channels, channel_size, num_per_sum,
        local_input.LockedBuffer(), local_input.LDim(),
        local_gradient_wrt_output.LockedBuffer(), local_gradient_wrt_output.LDim(),
        local_mean.LockedBuffer(), local_var.LockedBuffer(), this->m_epsilon,
        local_scale.LockedBuffer(),
        local_mean_gradient.LockedBuffer(), local_var_gradient.LockedBuffer(),
        local_gradient_wrt_input.Buffer(), local_gradient_wrt_input.LDim());
  }

}

#define PROTO(T)                                      \
  template class batch_normalization_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
