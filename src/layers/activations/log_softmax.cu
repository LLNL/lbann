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

#define LBANN_LOG_SOFTMAX_LAYER_INSTANTIATE
#include "lbann/comm_impl.hpp"
#include "lbann/layers/activations/log_softmax.hpp"
#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/softmax.hpp"
#endif // LBANN_HAS_DNN_LIB

namespace lbann {

namespace {

/** @brief Max functor */
template <class T>
struct max_op
{
  __device__ __forceinline__ DataType operator()(const T& x1, const T& x2) const
  {
    return gpu_lib::max(x1, x2);
  }
};

/** @brief Kernel for max reduction on matrix columns
 *
 *  Each CUDA block computes the max over a subset of matrix entries
 *  and outputs the result. This is repeated multiple times for
 *  column-wise max reduction.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimension: (height / bsize) x width x 1
 *
 *  @param values       (height x width) matrix
 *  @param max_values   (nblocksx x width) matrix
 */
template <size_t bsize, typename TensorDataType>
__global__ void reduce_max_kernel(size_t height,
                                  size_t width,
                                  const TensorDataType* __restrict__ values,
                                  size_t values_ldim,
                                  TensorDataType* __restrict__ max_values)
{
  // Indices
  const size_t tid = threadIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t bidx = blockIdx.x;
  const size_t bidy = blockIdx.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nblocksx = gridDim.x;
  const size_t nblocksy = gridDim.y;

  for (size_t col = bidy; col < width; col += nblocksy) {

    // Find largest value for each thread
    TensorDataType thread_max_val{-gpu_lib::infinity<TensorDataType>()};
    for (size_t row = gidx; row < height; row += nthreadsx) {
      const auto& val = values[row + col * values_ldim];
      thread_max_val = gpu_lib::max(thread_max_val, val);
    }

    // Find largest value for each block
    const TensorDataType block_max_val =
      gpu_lib::block_reduce<bsize, 1, 1, DataType, max_op<DataType>>(
        thread_max_val);
    if (tid == 0) {
      max_values[bidx + col * nblocksx] = block_max_val;
    }
  }
}

/** @brief Kernel for matrix column sums
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimension: (height / bsize) x width x 1
 *
 *  @param sums On input, array of zeros. On output, sum(x) for each
 *              column.
 */
template <size_t bsize, typename TensorDataType>
__global__ void reduce_sum_kernel(size_t height,
                                  size_t width,
                                  const TensorDataType* __restrict__ values,
                                  size_t values_ldim,
                                  TensorDataType* __restrict__ sums)
{

  // Indices
  const size_t tid = threadIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t bidy = blockIdx.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nblocksy = gridDim.y;

  for (size_t col = bidy; col < width; col += nblocksy) {

    // Compute sum for each thread
    TensorDataType thread_sum{0};
    for (size_t row = gidx; row < height; row += nthreadsx) {
      thread_sum += values[row + col * values_ldim];
    }

    // Compute sum for each block
    const TensorDataType block_sum =
      gpu_lib::block_reduce<bsize, 1, 1>(thread_sum);
    if (tid == 0) {
      gpu_lib::atomic_add(&sums[col], block_sum);
    }
  }
}

/** @brief Compute sum(exp(x-shift)) for each matrix column
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimension: (height / bsize) x width x 1
 *
 *  @param shifts   max(x) for each column
 *  @param sums     On input, array of zeros. On output,
 *                  sum(exp(x-shift)) for each column.
 */
template <size_t bsize, typename TensorDataType>
__global__ void fp_sumexp_kernel(size_t height,
                                 size_t width,
                                 const TensorDataType* __restrict__ input,
                                 size_t input_ldim,
                                 const TensorDataType* __restrict__ shifts,
                                 TensorDataType* __restrict__ sums)
{

  // Indices
  const size_t tid = threadIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t bidy = blockIdx.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nblocksy = gridDim.y;

  for (size_t col = bidy; col < width; col += nblocksy) {
    const auto& shift = shifts[col];

    // Exponentiate inputs and compute sum for each thread
    TensorDataType thread_sum{0};
    for (size_t row = gidx; row < height; row += nthreadsx) {
      const auto& x = input[row + col * input_ldim];
      thread_sum += gpu_lib::exp(x - shift);
    }

    // Compute sum for each block
    const TensorDataType block_sum =
      gpu_lib::block_reduce<bsize, 1, 1>(thread_sum);
    if (tid == 0) {
      gpu_lib::atomic_add(&sums[col], block_sum);
    }
  }
}

/** @brief Compute layer output
 *
 *  y = x - shift - log(sum(x-shift))
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimension: (height / bsize) x width x 1
 *
 *  @param shifts   max(x) for each column
 *  @param sums     sum(exp(x-shift)) for each column
 */
template <typename TensorDataType>
__global__ void fp_output_kernel(size_t height,
                                 size_t width,
                                 const TensorDataType* __restrict__ input,
                                 size_t input_ldim,
                                 TensorDataType* __restrict__ output,
                                 size_t output_ldim,
                                 const TensorDataType* __restrict__ shifts,
                                 const TensorDataType* __restrict__ sums)
{
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t col = gidy; col < width; col += nthreadsy) {
    const auto& shift = shifts[col];
    const TensorDataType log_sum_exp = gpu_lib::log(sums[col]);
    for (size_t row = gidx; row < height; row += nthreadsx) {
      const auto& x = input[row + col * input_ldim];
      auto& y = output[row + col * output_ldim];
      y = x - shift - log_sum_exp;
    }
  }
}

/** @brief Compute gradient w.r.t. input
 *
 *  dx = dy - softmax(x) * sum(dy)
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimension: (height / bsize) x width x 1
 *
 *  @param sums Column sums of the gradient w.r.t. output
 */
template <typename TensorDataType>
__global__ void
bp_kernel(size_t height,
          size_t width,
          const TensorDataType* __restrict__ output,
          size_t output_ldim,
          const TensorDataType* __restrict__ gradient_wrt_output,
          size_t gradient_wrt_output_ldim,
          const TensorDataType* __restrict__ sums,
          TensorDataType* __restrict__ gradient_wrt_input,
          size_t gradient_wrt_input_ldim)
{
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t col = gidy; col < width; col += nthreadsy) {
    const auto& sum = sums[col];
    for (size_t row = gidx; row < height; row += nthreadsx) {
      const auto& y = output[row + col * output_ldim];
      const auto& dy =
        gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
      dx = dy - gpu_lib::exp(y) * sum;
    }
  }
}

} // namespace

template <typename TensorDataType>
void fp_compute_impl(log_softmax_layer<TensorDataType,
                                       data_layout::DATA_PARALLEL,
                                       El::Device::GPU>& l)
{
  const TensorDataType zero = 0;
  const TensorDataType one = 1;
  const auto& local_input =
    dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(
      l.get_local_prev_activations());
  auto& local_output =
    dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(
      l.get_local_activations());
  dnn_lib::softmax_forward(one,
                           l.m_tensors_dnn_desc.get_prev_activations(),
                           local_input,
                           zero,
                           l.m_tensors_dnn_desc.get_activations(),
                           local_output,
                           softmax_mode::INSTANCE,
                           softmax_alg::LOG);
}

template <typename TensorDataType>
void bp_compute_impl(log_softmax_layer<TensorDataType,
                                       data_layout::DATA_PARALLEL,
                                       El::Device::GPU>& l)
{
  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;
  const TensorDataType zero = 0;
  const TensorDataType one = 1;
  const auto& local_output =
    dynamic_cast<const GPUMatType&>(l.get_local_activations());
  const auto& local_gradient_wrt_output =
    dynamic_cast<const GPUMatType&>(l.get_local_prev_error_signals());
  auto& local_gradient_wrt_input =
    dynamic_cast<GPUMatType&>(l.get_local_error_signals());
  dnn_lib::softmax_backward(one,
                            l.m_tensors_dnn_desc.get_activations(),
                            local_output,
                            l.m_tensors_dnn_desc.get_prev_error_signals(),
                            local_gradient_wrt_output,
                            zero,
                            l.m_tensors_dnn_desc.get_error_signals(),
                            local_gradient_wrt_input,
                            softmax_mode::INSTANCE,
                            softmax_alg::LOG);
}

template <typename TensorDataType>
void fp_compute_impl(log_softmax_layer<TensorDataType,
                                       data_layout::MODEL_PARALLEL,
                                       El::Device::GPU>& l)
{
  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;

  // Setup workspace
  l.m_workspace->Empty(false);
  l.m_workspace->AlignWith(l.get_activations());
  l.m_workspace->Resize(1, l.get_activations().Width());

  // Local matrices
  const auto& local_input =
    dynamic_cast<const GPUMatType&>(l.get_local_prev_activations());
  auto& local_output = dynamic_cast<GPUMatType&>(l.get_local_activations());
  auto& local_workspace = dynamic_cast<GPUMatType&>(l.m_workspace->Matrix());
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // GPU objects
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_input),
                                     gpu::get_sync_info(local_output),
                                     gpu::get_sync_info(local_workspace));
  // The comm templates will not convert the multisync, so cast the multisync
  // and use sync_info for comms.
  El::SyncInfo<El::Device::GPU> const& sync_info = multisync;

  // Find max value in each column
  gpu_lib::thrust::vector<TensorDataType> max_vals;
  if (local_input.IsEmpty()) {
    max_vals.resize(local_width, -std::numeric_limits<DataType>::infinity());
  }
  else {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    max_vals.resize(grid_dims.x * local_width);

    // Launch GPU Kernel
    hydrogen::gpu::LaunchKernel(reduce_max_kernel<block_size, TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                local_height,
                                local_width,
                                local_input.LockedBuffer(),
                                local_input.LDim(),
                                max_vals.data().get());
    while (grid_dims.x > 1) {
      const size_t prev_height = grid_dims.x;
      grid_dims.x = (prev_height + block_size - 1) / block_size;
      gpu_lib::thrust::vector<TensorDataType> prev_vals(std::move(max_vals));
      max_vals.resize(grid_dims.x * local_width);
      hydrogen::gpu::LaunchKernel(reduce_max_kernel<block_size, TensorDataType>,
                                  grid_dims,
                                  block_dims,
                                  0,
                                  multisync,
                                  prev_height,
                                  local_width,
                                  prev_vals.data().get(),
                                  prev_height,
                                  max_vals.data().get());
    }
  }
  El::mpi::AllReduce(max_vals.data().get(),
                     max_vals.size(),
                     El::mpi::MAX,
                     l.m_workspace->RedundantComm(),
                     sync_info);

  // Compute sum(exp(x-max_val)) for each column
  El::Zero(*l.m_workspace);
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    hydrogen::gpu::LaunchKernel(fp_sumexp_kernel<block_size, TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                local_height,
                                local_width,
                                local_input.LockedBuffer(),
                                local_input.LDim(),
                                max_vals.data().get(),
                                local_workspace.Buffer());
  }
  l.get_comm()->allreduce(*l.m_workspace, l.m_workspace->RedundantComm());

  // Compute output
  // Note: y = x - max_val - log(sum(exp(x-max_val)))
  if (!local_output.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    hydrogen::gpu::LaunchKernel(fp_output_kernel<TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                local_height,
                                local_width,
                                local_input.LockedBuffer(),
                                local_input.LDim(),
                                local_output.Buffer(),
                                local_output.LDim(),
                                max_vals.data().get(),
                                local_workspace.LockedBuffer());
  }
}

template <typename TensorDataType>
void bp_compute_impl(log_softmax_layer<TensorDataType,
                                       data_layout::MODEL_PARALLEL,
                                       El::Device::GPU>& l)
{
  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;
  // Local matrices
  const auto& local_output =
    dynamic_cast<const GPUMatType&>(l.get_local_activations());
  const auto& local_gradient_wrt_output =
    dynamic_cast<const GPUMatType&>(l.get_local_prev_error_signals());
  auto& local_gradient_wrt_input =
    dynamic_cast<GPUMatType&>(l.get_local_error_signals());
  auto& local_workspace = dynamic_cast<GPUMatType&>(l.m_workspace->Matrix());
  const auto& local_height = local_output.Height();
  const auto& local_width = local_output.Width();

  // GPU objects
  auto multisync =
    El::MakeMultiSync(gpu::get_sync_info(local_output),
                      gpu::get_sync_info(local_gradient_wrt_output),
                      gpu::get_sync_info(local_gradient_wrt_input),
                      gpu::get_sync_info(local_workspace));

  // Compute sum of entries in gradient w.r.t. output
  El::Zero(local_workspace);
  if (!local_gradient_wrt_output.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    hydrogen::gpu::LaunchKernel(reduce_sum_kernel<block_size, TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                local_height,
                                local_width,
                                local_gradient_wrt_output.LockedBuffer(),
                                local_gradient_wrt_output.LDim(),
                                local_workspace.Buffer());
  }
  l.get_comm()->allreduce(*l.m_workspace, l.m_workspace->RedundantComm());

  // Compute gradient w.r.t. input
  if (!local_gradient_wrt_input.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    hydrogen::gpu::LaunchKernel(bp_kernel<TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                local_height,
                                local_width,
                                local_output.LockedBuffer(),
                                local_output.LDim(),
                                local_gradient_wrt_output.LockedBuffer(),
                                local_gradient_wrt_output.LDim(),
                                local_workspace.LockedBuffer(),
                                local_gradient_wrt_input.Buffer(),
                                local_gradient_wrt_input.LDim());
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void log_softmax_layer<TensorDataType, Layout, Device>::fp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("log_softmax_layer::fp_compute");
  fp_compute_impl(*this);
}
template <typename TensorDataType, data_layout Layout, El::Device Device>
void log_softmax_layer<TensorDataType, Layout, Device>::bp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("log_softmax_layer::bp_compute");
  bp_compute_impl(*this);
}

// Template instantiation
#define PROTO(T)                                                               \
  template class log_softmax_layer<T,                                          \
                                   data_layout::DATA_PARALLEL,                 \
                                   El::Device::GPU>;                           \
  template class log_softmax_layer<T,                                          \
                                   data_layout::MODEL_PARALLEL,                \
                                   El::Device::GPU>;

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
