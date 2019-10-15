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

#define LBANN_SOFTMAX_LAYER_INSTANTIATE
#include "lbann/layers/activations/softmax.hpp"
#include "lbann/utils/cuda.hpp"

namespace lbann {

namespace {

<<<<<<< HEAD
#ifdef LBANN_ENABLE_SOFTMAX_THRESHOLD
/** Functor to ensure values are above threshold value */
struct threshold_op {
  __forceinline__ __device__ DataType operator()(const DataType& y) const {
    return cuda::max(y, cuda::sqrt(cuda::min<TensorDataType>()));
  }
};
#endif // LBANN_ENABLE_SOFTMAX_THRESHOLD

/** @brief Max functor */
template <class T>
struct max_op {
  __device__ __forceinline__
  DataType operator()(const T& x1, const T& x2) const {
    return cuda::max(x1, x2);
=======
/** Minimum output value to avoid denormalized floats. */
inline __device__ TensorDataType get_min_output() {
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
  return cuda::sqrt(cuda::min<TensorDataType>());
#else
  return TensorDataType(0);
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF
}

#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
/** Operator for thresholding output. */
struct fp_threshold_op {
  const TensorDataType min_output =this->get_min_output();
  inline __device__ TensorDataType operator()(const TensorDataType& y) const {
    return cuda::max(y, min_output);
  }
};
/** Operator for thresholding gradient w.r.t. input. */
struct bp_threshold_op {
  const TensorDataType min_output =this->get_min_output();
  inline __device__ TensorDataType operator()(const TensorDataType& y,
                                        const TensorDataType& dx) const {
    return (y > min_output) ? dx : TensorDataType(0);
>>>>>>> Working on the math and activations layers.
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
<<<<<<< HEAD
template <size_t bsize>
__global__ void reduce_max_kernel(size_t height,
                                  size_t width,
                                  const DataType* __restrict__ values,
                                  size_t values_ldim,
                                  DataType* __restrict__ max_values) {
=======
template <El::Int block_size>
__global__ void reduce_max_kernel(El::Int height, El::Int width,
                                  const TensorDataType* __restrict__ values,
                                  El::Int values_ldim,
                                  TensorDataType* __restrict__ max_values) {
>>>>>>> Working on the math and activations layers.

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
<<<<<<< HEAD
    DataType thread_max_val{-cuda::infinity<DataType>()};
    for (size_t row = gidx; row < height; row += nthreadsx) {
      const auto& val = values[row+col*values_ldim];
      thread_max_val = cuda::max(thread_max_val, val);
    }

    // Find largest value for each block
    const DataType block_max_val
      = cuda::block_reduce<bsize,1,1,DataType,max_op<DataType>>(thread_max_val);
=======
    TensorDataType private_max_val = -cuda::infinity<TensorDataType>();
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      private_max_val = cuda::max(private_max_val,
                                  values[row + col * values_ldim]);
    }

    // Shared memory reduction to get largest value for each block
    __shared__ TensorDataType shared_max_vals[block_size];
    shared_max_vals[tid] = private_max_val;
    for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_max_vals[tid] = cuda::max(shared_max_vals[tid],
                                         shared_max_vals[tid + stride]);
      }
    }
>>>>>>> Working on the math and activations layers.
    if (tid == 0) {
      max_values[bidx+col*nblocksx] = block_max_val;
    }

  }

}

/** @brief Compute exp(x-shift)
 *
 *  Also compute sum(exp(x-shift)) for each matrix column.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimension: (height / bsize) x width x 1
 */
<<<<<<< HEAD
template <size_t bsize>
__global__ void fp_exp_kernel(size_t height,
                              size_t width,
                              const DataType* __restrict__ input,
                              size_t input_ldim,
                              DataType* __restrict__ output,
                              size_t output_ldim,
                              const DataType* __restrict__ shifts,
                              DataType* __restrict__ sums) {

  // Indices
  const size_t tid = threadIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t bidy = blockIdx.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nblocksy = gridDim.y;

  for (size_t col = bidy; col < width; col += nblocksy) {
    const auto& shift = shifts[col];

    // Exponentiate inputs and compute sum for each thread
    DataType thread_sum{0};
    for (size_t row = gidx; row < height; row += nthreadsx) {
      const auto& x = input[row+col*input_ldim];
      auto& y = output[row+col*output_ldim];
      y = cuda::exp(x-shift);
      thread_sum += y;
=======
template <El::Int block_size>
__global__ void fp_exp_kernel(El::Int height, El::Int width,
                              const TensorDataType* __restrict__ input,
                              El::Int input_ldim,
                              TensorDataType* __restrict__ output,
                              El::Int output_ldim,
                              const TensorDataType* __restrict__ shifts,
                              El::Int shifts_stride,
                              TensorDataType* __restrict__ sums,
                              El::Int sums_stride) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nblocksy = gridDim.y;

  // Reduce each matrix column independently
  for (El::Int col = bidy; col < width; col += nblocksy) {
    const auto& shift = shifts[col * shifts_stride];

    // Exponentiate and compute sum for each thread
    TensorDataType private_sum = 0;
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      const auto& x = input[row + col * input_ldim];
      auto& y = output[row + col * output_ldim];
      y = cuda::exp(x - shift);
      private_sum += y;
    }

    // Shared memory reduction to get sum for each block
    __shared__ TensorDataType shared_sums[block_size];
    shared_sums[tid] = private_sum;
    for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_sums[tid] += shared_sums[tid + stride];
      }
>>>>>>> Working on the math and activations layers.
    }

    // Compute sum for each block
    const DataType block_sum = cuda::block_reduce<bsize,1,1>(thread_sum);
    if (tid == 0) {
      cuda::atomic_add(&sums[col], block_sum);
    }

  }

}

/** @brief Compute layer output
 *
 *  y = exp(x-shift) / sum(exp(x-shift))
 *
 *  If @c LBANN_ENABLE_SOFTMAX_THRESHOLD is set, small values are
 *  thresholded to a minimum value to avoid denormalized floats.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimension: (height / bsize) x width x 1
 *
 *  @param output   On input, constains exp(x-shift). On output,
 *                  contains the layer output.
 *  @param sums     sum(exp(x-shift)) for each column
 */
<<<<<<< HEAD
__global__ void fp_output_kernel(size_t height,
                                 size_t width,
                                 DataType* __restrict__ output,
                                 size_t output_ldim,
                                 const DataType* __restrict__ sums) {
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t col = gidy; col < width; col += nthreadsy) {
    const auto& denom = sums[col];
    for (size_t row = gidx; row < height; row += nthreadsx) {
      auto& y = output[row+col*output_ldim];
      y /= denom;
#ifdef LBANN_ENABLE_SOFTMAX_THRESHOLD
      y = cuda::max(y, cuda::sqrt(cuda::min<DataType>()));
#endif // LBANN_ENABLE_SOFTMAX_THRESHOLD
=======
__global__ void fp_scale_kernel(El::Int height, El::Int width,
                                TensorDataType* __restrict__ output,
                                El::Int output_ldim,
                                const TensorDataType* __restrict__ sums,
                                El::Int sums_stride) {
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nblocksy = gridDim.y;
  const auto& min_output =this->get_min_output();
  for (El::Int col = bidy; col < width; col += nblocksy) {
    const auto& scale = 1 / sums[col * sums_stride];
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      auto& y = output[row + col * output_ldim];
      y = cuda::max(scale * y, min_output);
>>>>>>> Working on the math and activations layers.
    }
  }
}

<<<<<<< HEAD
/** @brief Compute dot(y,dy) for each matrix column
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimension: (height / bsize) x width x 1
 */
template <size_t bsize>
__global__ void bp_dot_product_kernel(size_t height,
                                      size_t width,
                                      const DataType* __restrict__ output,
                                      size_t output_ldim,
                                      const DataType* __restrict__ gradient_wrt_output,
                                      size_t gradient_wrt_output_ldim,
                                      DataType* __restrict__ dot_products) {
=======
/** Compute dot products between output and gradient w.r.t. output. */
template <El::Int block_size>
__global__ void bp_dot_product_kernel(El::Int height, El::Int width,
                                      const TensorDataType* __restrict__ output,
                                      El::Int output_ldim,
                                      const TensorDataType* __restrict__ gradient_wrt_output,
                                      El::Int gradient_wrt_output_ldim,
                                      TensorDataType* __restrict__ dot_products,
                                      El::Int dot_products_stride) {
>>>>>>> Working on the math and activations layers.

  // Indices
  const size_t tid = threadIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t bidy = blockIdx.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nblocksy = gridDim.y;

  for (size_t col = bidy; col < width; col += nblocksy) {

    // Compute dot product contribution for each thread
<<<<<<< HEAD
    DataType thread_dot_product{0};
    for (size_t row = gidx; row < height; row += nthreadsx) {
      const auto& y = output[row+col*output_ldim];
      const auto& dy = gradient_wrt_output[row+col*gradient_wrt_output_ldim];
      thread_dot_product += y * dy;
=======
    TensorDataType private_dot_product = 0;
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      const auto& y = output[row + col * output_ldim];
      const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      private_dot_product += y * dy;
    }

    // Shared memory reduction to get contribution for each block
    __shared__ TensorDataType shared_dot_products[block_size];
    shared_dot_products[tid] = private_dot_product;
    for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_dot_products[tid] += shared_dot_products[tid + stride];
      }
>>>>>>> Working on the math and activations layers.
    }

    // Compute dot product contribution for each block
    const DataType block_dot_product
      = cuda::block_reduce<bsize,1,1>(thread_dot_product);
    if (tid == 0) {
      cuda::atomic_add(&dot_products[col], block_dot_product);
    }

  }

}

<<<<<<< HEAD
/** @brief Compute gradient w.r.t. input
 *
 *  dx = y * (dy - dot(y,dy))
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimension: (height / bsize) x width x 1
 *
 *  @param dot_products dot(y,dy) for each matrix column
 */
template <size_t bsize>
__global__ void bp_kernel(size_t height,
                          size_t width,
                          const DataType* __restrict__ output,
                          size_t output_ldim,
                          const DataType* __restrict__ gradient_wrt_output,
                          size_t gradient_wrt_output_ldim,
                          const DataType* __restrict__ dot_products,
                          DataType* __restrict__ gradient_wrt_input,
                          size_t gradient_wrt_input_ldim) {
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t col = gidy; col < width; col += nthreadsy) {
    const auto& y_dot_dy = dot_products[col];
    for (size_t row = gidx; row < height; row += nthreadsx) {
      const auto& y = output[row+col*output_ldim];
      const auto& dy = gradient_wrt_output[row+col*gradient_wrt_output_ldim];
      auto& dx = gradient_wrt_input[row+col*gradient_wrt_input_ldim];
      dx = y * (dy - y_dot_dy);
=======
/** Compute gradient w.r.t. input. */
template <El::Int block_size>
__global__ void bp_kernel(El::Int height, El::Int width,
                          const TensorDataType* __restrict__ output,
                          El::Int output_ldim,
                          const TensorDataType* __restrict__ gradient_wrt_output,
                          El::Int gradient_wrt_output_ldim,
                          const TensorDataType* __restrict__ dot_products,
                          El::Int dot_products_stride,
                          TensorDataType* __restrict__ gradient_wrt_input,
                          El::Int gradient_wrt_input_ldim) {
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nblocksy = gridDim.y;
  const auto& min_output =this->get_min_output();
  for (El::Int col = bidy; col < width; col += nblocksy) {
    const auto& y_dot_dy = dot_products[col * dot_products_stride];
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      const auto& y = output[row + col * output_ldim];
      const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
      dx = (y > min_output) ? y * (dy - y_dot_dy) : TensorDataType(0);
>>>>>>> Working on the math and activations layers.
    }
  }
}

} // namespace

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
  constexpr TensorDataType zero = 0;
  constexpr TensorDataType one = 1;
  const auto& local_input = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(this->get_local_activations());
  if (!local_input.IsEmpty()) {
    CHECK_CUDNN(cudnnSoftmaxForward(cudnn::get_handle(),
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &one,
                                    m_tensors_cudnn_desc.get_prev_activations(),
                                    local_input.LockedBuffer(),
                                    &zero,
                                    m_tensors_cudnn_desc.get_activations(),
                                    local_output.Buffer()));
#ifdef LBANN_ENABLE_SOFTMAX_THRESHOLD
    cuda::apply_entrywise_unary_operator<threshold_op>(local_output,
                                                       local_output);
#endif // LBANN_ENABLE_SOFTMAX_THRESHOLD
  }
}

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
  constexpr TensorDataType zero = 0;
  constexpr TensorDataType one = 1;
  const auto& local_output = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(this->get_local_activations());
  const auto& local_gradient_wrt_output = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(this->get_local_prev_error_signals());
  auto& local_gradient_wrt_input = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(this->get_local_error_signals());
  if (!local_output.IsEmpty()) {
    CHECK_CUDNN(cudnnSoftmaxBackward(cudnn::get_handle(),
                                     CUDNN_SOFTMAX_ACCURATE,
                                     CUDNN_SOFTMAX_MODE_INSTANCE,
                                     &one,
                                     m_tensors_cudnn_desc.get_activations(),
                                     local_output.LockedBuffer(),
                                     m_tensors_cudnn_desc.get_prev_error_signals(),
                                     local_gradient_wrt_output.LockedBuffer(),
                                     &zero,
                                     m_tensors_cudnn_desc.get_error_signals(),
                                     local_gradient_wrt_input.Buffer()));
  }
}

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {

  // Local matrices
  const auto& local_input = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(this->get_local_activations());
  auto& local_workspace = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(m_workspace->Matrix());
  const size_t local_height = local_input.Height();
  const size_t local_width = local_input.Width();

  // GPU objects
  auto&& stream = El::GPUManager::Stream();
  auto&& event = El::GPUManager::Event();
  El::SyncInfo<El::Device::GPU> sync_info{stream, event};

  // Find max value in each column
  cuda::thrust::vector<TensorDataType> max_vals;
  if (local_output.IsEmpty()) {
    max_vals.resize(local_width,
                    -std::numeric_limits<TensorDataType>::infinity());
  }
  else {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    max_vals.resize(grid_dims.x * local_width);
    reduce_max_kernel<block_size><<<grid_dims, block_dims, 0, stream>>>(
      local_height, local_width,
      local_input.LockedBuffer(), local_input.LDim(),
      max_vals.data().get());
    while (grid_dims.x > 1) {
      const size_t prev_height = grid_dims.x;
      grid_dims.x = (prev_height + block_size - 1) / block_size;
      cuda::thrust::vector<DataType> prev_vals(std::move(max_vals));
      max_vals.resize(grid_dims.x * local_width);
      reduce_max_kernel<block_size><<<grid_dims, block_dims, 0, stream>>>(
        prev_height, local_width,
        prev_vals.data().get(), prev_height,
        max_vals.data().get());
    }
  }
  El::mpi::AllReduce(max_vals.data().get(), max_vals.size(),
                     El::mpi::MAX, m_workspace->RedundantComm(),
                     sync_info);

  // Compute exp(x-max_val) and sum(exp(x-max_val))
  El::Zero(*m_workspace);
  if (!local_output.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    fp_exp_kernel<block_size><<<grid_dims, block_dims, 0, stream>>>(
      local_height, local_width,
      local_input.LockedBuffer(), local_input.LDim(),
      local_output.Buffer(), local_output.LDim(),
      max_vals.data().get(),
      local_workspace.Buffer());
  }
  El::AllReduce(*m_workspace, m_workspace->RedundantComm());

  // Compute output
  // Note: y = exp(x-max_val) / sum(exp(x-max_val))
  if (!local_output.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    fp_output_kernel<<<grid_dims, block_dims, 0, stream>>>(
      local_height, local_width,
      local_output.Buffer(), local_output.LDim(),
      local_workspace.LockedBuffer());
  }

}

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {

  // Local matrices
  const auto& local_output = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(this->get_local_activations());
  const auto& local_gradient_wrt_output = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(this->get_local_prev_error_signals());
  auto& local_gradient_wrt_input = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(this->get_local_error_signals());
  auto& local_workspace = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(m_workspace->Matrix());
  const auto& local_height = local_output.Height();
  const auto& local_width = local_output.Width();

  // GPU objects
  auto&& stream = El::GPUManager::Stream();
  auto&& event = El::GPUManager::Event();
  El::SyncInfo<El::Device::GPU> sync_info{stream, event};

  // Compute dot(y,dy)
  El::Zero(local_workspace);
  if (!local_output.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    bp_dot_product_kernel<block_size>
      <<<grid_dims, block_dims, 0, stream>>>(
        local_height, local_width,
        local_output.LockedBuffer(),
        local_output.LDim(),
        local_gradient_wrt_output.LockedBuffer(),
        local_gradient_wrt_output.LDim(),
        local_workspace.Buffer());
  }
  El::AllReduce(*m_workspace, m_workspace->RedundantComm());

  // Compute gradient w.r.t. input
  if (!local_output.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    bp_kernel<block_size><<<grid_dims, block_dims, 0, stream>>>(
      local_height, local_width,
      local_output.LockedBuffer(),
      local_output.LDim(),
      local_gradient_wrt_output.LockedBuffer(),
      local_gradient_wrt_output.LDim(),
      local_workspace.Buffer(),
      local_gradient_wrt_input.Buffer(),
      local_gradient_wrt_input.LDim());
  }

}

// Template instantiation
template class softmax_layer<
  float, data_layout::DATA_PARALLEL, El::Device::GPU>;
template class softmax_layer<
  float, data_layout::MODEL_PARALLEL, El::Device::GPU>;

} // namespace lbann
