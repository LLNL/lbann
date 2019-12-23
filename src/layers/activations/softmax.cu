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

namespace lbann {

namespace {

/** Minimum output value to avoid denormalized floats. */
inline __device__ DataType get_min_output() {
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
  return cuda::sqrt(cuda::min<DataType>());
#else
  return DataType(0);
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF
}

#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
/** Operator for thresholding output. */
struct fp_threshold_op {
  const DataType min_output = get_min_output();
  inline __device__ DataType operator()(const DataType& y) const {
    return cuda::max(y, min_output);
  }
};
/** Operator for thresholding gradient w.r.t. input. */
struct bp_threshold_op {
  const DataType min_output = get_min_output();
  inline __device__ DataType operator()(const DataType& y,
                                        const DataType& dx) const {
    return (y > min_output) ? dx : DataType(0);
  }
};
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF

/** Find largest entry within each CUDA block.
 *  Each block is assigned several entries from the same mini-batch
 *  sample and it finds the largest entry. Results are output to an
 *  nblocksx x width matrix.
 */
template <El::Int block_size>
__global__ void reduce_max_kernel(El::Int height, El::Int width,
                                  const DataType* __restrict__ values,
                                  El::Int values_ldim,
                                  DataType* __restrict__ max_values) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidx = blockIdx.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nblocksx = gridDim.x;
  const El::Int nblocksy = gridDim.y;

  // Reduce each matrix column independently
  for (El::Int col = bidy; col < width; col += nblocksy) {

    // Find largest value for each thread
    DataType private_max_val = -cuda::infinity<DataType>();
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      private_max_val = cuda::max(private_max_val,
                                  values[row + col * values_ldim]);
    }

    // Shared memory reduction to get largest value for each block
    __shared__ DataType shared_max_vals[block_size];
    shared_max_vals[tid] = private_max_val;
    for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_max_vals[tid] = cuda::max(shared_max_vals[tid],
                                         shared_max_vals[tid + stride]);
      }
    }
    if (tid == 0) {
      max_values[bidx + col*nblocksx] = shared_max_vals[0];
    }

  }

}

/** Exponentiate outputs and compute column sums.
 *  Subtracting by the column max prevents output from blowing
 *  up. Large negative values underflow to 0.
 */
template <El::Int block_size>
__global__ void fp_exp_kernel(El::Int height, El::Int width,
                              const DataType* __restrict__ input,
                              El::Int input_ldim,
                              DataType* __restrict__ output,
                              El::Int output_ldim,
                              const DataType* __restrict__ shifts,
                              El::Int shifts_stride,
                              DataType* __restrict__ sums,
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
    DataType private_sum = 0;
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      const auto& x = input[row + col * input_ldim];
      auto& y = output[row + col * output_ldim];
      y = cuda::exp(x - shift);
      private_sum += y;
    }

    // Shared memory reduction to get sum for each block
    __shared__ DataType shared_sums[block_size];
    shared_sums[tid] = private_sum;
    for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_sums[tid] += shared_sums[tid + stride];
      }
    }

    // Atomic add to global sum
    if (tid == 0) {
      cuda::atomic_add(&sums[col * sums_stride], shared_sums[0]);
    }

  }

}

/** Divide outputs by column sums.
 *  Small values can be rounded to minimum output value to avoid
 *  denormalized floats.
 */
__global__ void fp_scale_kernel(El::Int height, El::Int width,
                                DataType* __restrict__ output,
                                El::Int output_ldim,
                                const DataType* __restrict__ sums,
                                El::Int sums_stride) {
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nblocksy = gridDim.y;
  const auto& min_output = get_min_output();
  for (El::Int col = bidy; col < width; col += nblocksy) {
    const auto& scale = 1 / sums[col * sums_stride];
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      auto& y = output[row + col * output_ldim];
      y = cuda::max(scale * y, min_output);
    }
  }
}

/** Compute dot products between output and gradient w.r.t. output. */
template <El::Int block_size>
__global__ void bp_dot_product_kernel(El::Int height, El::Int width,
                                      const DataType* __restrict__ output,
                                      El::Int output_ldim,
                                      const DataType* __restrict__ gradient_wrt_output,
                                      El::Int gradient_wrt_output_ldim,
                                      DataType* __restrict__ dot_products,
                                      El::Int dot_products_stride) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nblocksy = gridDim.y;

  // Compute dot product for each matrix column independently
  for (El::Int col = bidy; col < width; col += nblocksy) {

    // Compute dot product contribution for each thread
    DataType private_dot_product = 0;
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      const auto& y = output[row + col * output_ldim];
      const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      private_dot_product += y * dy;
    }

    // Shared memory reduction to get contribution for each block
    __shared__ DataType shared_dot_products[block_size];
    shared_dot_products[tid] = private_dot_product;
    for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_dot_products[tid] += shared_dot_products[tid + stride];
      }
    }

    // Atomic add to global dot product
    if (tid == 0) {
      cuda::atomic_add(&dot_products[col * dot_products_stride],
                       shared_dot_products[0]);
    }

  }

}

/** Compute gradient w.r.t. input. */
template <El::Int block_size>
__global__ void bp_kernel(El::Int height, El::Int width,
                          const DataType* __restrict__ output,
                          El::Int output_ldim,
                          const DataType* __restrict__ gradient_wrt_output,
                          El::Int gradient_wrt_output_ldim,
                          const DataType* __restrict__ dot_products,
                          El::Int dot_products_stride,
                          DataType* __restrict__ gradient_wrt_input,
                          El::Int gradient_wrt_input_ldim) {
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nblocksy = gridDim.y;
  const auto& min_output = get_min_output();
  for (El::Int col = bidy; col < width; col += nblocksy) {
    const auto& y_dot_dy = dot_products[col * dot_products_stride];
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      const auto& y = output[row + col * output_ldim];
      const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
      dx = (y > min_output) ? y * (dy - y_dot_dy) : DataType(0);
    }
  }
}

} // namespace

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
  constexpr DataType zero = 0;
  constexpr DataType one = 1;
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();

  cudnnSoftmaxMode_t softmax_mode;
  switch(m_mode) {
    case softmax_mode::INSTANCE:
      softmax_mode = CUDNN_SOFTMAX_MODE_INSTANCE;
      break;
    case softmax_mode::CHANNEL:
      softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
      break;
    default:
      LBANN_ERROR("Unsupported softmax mode");
  }

  if (!local_input.IsEmpty()) {
    CHECK_CUDNN(cudnnSoftmaxForward(cudnn::get_handle(),
                                    CUDNN_SOFTMAX_ACCURATE,
                                    softmax_mode,
                                    &one,
                                    m_tensors_cudnn_desc.get_prev_activations(),
                                    local_input.LockedBuffer(),
                                    &zero,
                                    m_tensors_cudnn_desc.get_activations(),
                                    local_output.Buffer()));
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
    cuda::apply_entrywise_unary_operator<fp_threshold_op>(local_output,
                                                          local_output);
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF
  }
}

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
  constexpr DataType zero = 0;
  constexpr DataType one = 1;
  const auto& local_output = get_local_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();

  cudnnSoftmaxMode_t softmax_mode;
  switch(m_mode) {
    case softmax_mode::INSTANCE:
      softmax_mode = CUDNN_SOFTMAX_MODE_INSTANCE;
      break;
    case softmax_mode::CHANNEL:
      softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
      break;
    default:
      LBANN_ERROR("Unsupported softmax mode");
  }

  if (!local_output.IsEmpty()) {
    CHECK_CUDNN(cudnnSoftmaxBackward(cudnn::get_handle(),
                                     CUDNN_SOFTMAX_ACCURATE,
                                     softmax_mode,
                                     &one,
                                     m_tensors_cudnn_desc.get_activations(),
                                     local_output.LockedBuffer(),
                                     m_tensors_cudnn_desc.get_prev_error_signals(),
                                     local_gradient_wrt_output.LockedBuffer(),
                                     &zero,
                                     m_tensors_cudnn_desc.get_error_signals(),
                                     local_gradient_wrt_input.Buffer()));
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
    cuda::apply_entrywise_binary_operator<bp_threshold_op>(local_output,
                                                           local_gradient_wrt_input,
                                                           local_gradient_wrt_input);
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF
  }
}

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {

  if(m_mode != softmax_mode::INSTANCE) {
    LBANN_ERROR("Unsupported softmax mode");
  }

  // Local matrices
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  auto& local_workspace = m_workspace->Matrix();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // GPU objects
  auto&& stream = El::GPUManager::Stream();
  auto&& event = El::GPUManager::Event();
  El::SyncInfo<El::Device::GPU> sync_info{stream, event};

  // Initialize CUDA threads/blocks
  // Note: kernels use a 2D thread distribution with a 256 x 1 block
  // and nblocksx x local_width grid.
  constexpr El::Int block_size = 256;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.y = local_width;

  // Find column-wise maximum entries
  grid_dims.x = (local_height + block_size - 1) / block_size;
  if (grid_dims.x < 1) { grid_dims.x = 1; }
  cuda::thrust::vector<DataType> max_vals(grid_dims.x * local_width);
  reduce_max_kernel<block_size><<<grid_dims, block_dims, 0, stream>>>(
    local_height, local_width,
    local_input.LockedBuffer(), local_input.LDim(),
    max_vals.data().get());
  while (grid_dims.x > 1) {
    const El::Int prev_height = grid_dims.x;
    grid_dims.x = (prev_height + block_size - 1) / block_size;
    cuda::thrust::vector<DataType> prev_vals(std::move(max_vals));
    max_vals.resize(grid_dims.x * local_width);
    reduce_max_kernel<block_size><<<grid_dims, block_dims, 0, stream>>>(
      prev_height, local_width,
      prev_vals.data().get(), prev_height,
      max_vals.data().get());
  }
  El::mpi::AllReduce(max_vals.data().get(), max_vals.size(),
                     El::mpi::MAX, m_workspace->RedundantComm(),
                     sync_info);

  // Exponentiate outputs and compute column sums
  El::Zero(*m_workspace);
  if (!local_output.IsEmpty()) {
    grid_dims.x = (local_height + block_size - 1) / block_size;
    fp_exp_kernel<block_size><<<grid_dims, block_dims, 0, stream>>>(
      local_height, local_width,
      local_input.LockedBuffer(), local_input.LDim(),
      local_output.Buffer(), local_output.LDim(),
      max_vals.data().get(), 1,
      local_workspace.Buffer(), 1);
  }
  El::AllReduce(*m_workspace, m_workspace->RedundantComm());

  // Divide activations by column sums
  if (!local_output.IsEmpty()) {
    grid_dims.x = (local_height + block_size - 1) / block_size;
    fp_scale_kernel<<<grid_dims, block_dims, 0, stream>>>(
      local_height, local_width,
      local_output.Buffer(), local_output.LDim(),
      local_workspace.LockedBuffer(), 1);
  }

}

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {

  if(m_mode != softmax_mode::INSTANCE) {
    LBANN_ERROR("Unsupported softmax mode");
  }

  // Local matrices
  const auto& local_output = get_local_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();
  auto& local_workspace = m_workspace->Matrix();
  const auto& local_height = local_output.Height();
  const auto& local_width = local_output.Width();

  // GPU objects
  auto&& stream = El::GPUManager::Stream();
  auto&& event = El::GPUManager::Event();
  El::SyncInfo<El::Device::GPU> sync_info{stream, event};

  // Initialize CUDA threads/blocks
  // Note: kernels use a 2D thread distribution with a 256 x 1 block
  // and nblocksx x local_width grid.
  constexpr El::Int block_size = 256;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.y = local_width;

  // Compute dot products between output and gradient w.r.t. output
  El::Zero(local_workspace);
  if (!local_output.IsEmpty()) {
    grid_dims.x = (local_height + block_size - 1) / block_size;
    bp_dot_product_kernel<block_size>
      <<<grid_dims, block_dims, 0, stream>>>(
        local_height, local_width,
        local_output.LockedBuffer(),
        local_output.LDim(),
        local_gradient_wrt_output.LockedBuffer(),
        local_gradient_wrt_output.LDim(),
        local_workspace.Buffer(), 1);
  }
  El::AllReduce(*m_workspace, m_workspace->RedundantComm());

  // Compute gradient w.r.t. input
  if (!local_output.IsEmpty()) {
    grid_dims.x = (local_height + block_size - 1) / block_size;
    bp_kernel<block_size><<<grid_dims, block_dims, 0, stream>>>(
      local_height, local_width,
      local_output.LockedBuffer(),
      local_output.LDim(),
      local_gradient_wrt_output.LockedBuffer(),
      local_gradient_wrt_output.LDim(),
      local_workspace.Buffer(), 1,
      local_gradient_wrt_input.Buffer(),
      local_gradient_wrt_input.LDim());
  }

}

template class softmax_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
template class softmax_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;

} // namespace lbann
