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

#include "lbann/layers/misc/covariance.hpp"

namespace lbann {

namespace {

/** Compute local contributions to means.
 *  Computes column-wise sums of two input matrices and multiplies
 *  them by a scaling factor (which should be
 *  1/height). 'contribution' is interpreted as a 2 x width matrix
 *  where the first row corresponds to 'input0' and the second row to
 *  'input1'.
 */
template <El::Int block_size>
__global__ void mean_contribution_kernel(El::Int height,
                                         El::Int width,
                                         DataType scale,
                                         const DataType* __restrict__ input0,
                                         El::Int input0_ldim,
                                         const DataType* __restrict__ input1,
                                         El::Int input1_ldim,
                                         DataType* __restrict__ contribution) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;

  // Compute local contribution for each matrix column
  for (El::Int col = bidy; col < width; col += gridDim.y) {

    // Compute contributions for each thread
    DataType private_contribution0 = 0;
    DataType private_contribution1 = 0;
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      private_contribution0 += input0[row + col * input0_ldim];
      private_contribution1 += input1[row + col * input1_ldim];
    }

    // Shared memory reduction to get contribution for each block
    /// @todo unroll loops
    __shared__ DataType shared_contribution0[block_size];
    __shared__ DataType shared_contribution1[block_size];
    shared_contribution0[tid] = private_contribution0;
    shared_contribution1[tid] = private_contribution1;
    for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_contribution0[tid] += shared_contribution0[tid + stride];
        shared_contribution1[tid] += shared_contribution1[tid + stride];
      }
    }
    if (tid == 0) {
      cuda::atomic_add(&contribution[2*col],
                       scale * shared_contribution0[0]);
      cuda::atomic_add(&contribution[2*col+1],
                       scale * shared_contribution1[0]);
    }

  }

}

/** Compute local contributions to covariances. */
template <El::Int block_size>
__global__ void covariance_contribution_kernel(El::Int height,
                                               El::Int width,
                                               DataType scale,
                                               const DataType* __restrict__ input0,
                                               El::Int input0_ldim,
                                               const DataType* __restrict__ input1,
                                               El::Int input1_ldim,
                                               const DataType* __restrict__ means,
                                               DataType* __restrict__ contribution) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;

  // Compute local contribution for each matrix column
  for (El::Int col = bidy; col < width; col += gridDim.y) {
    const auto& mean0 = means[2*col];
    const auto& mean1 = means[2*col+1];

    // Compute contributions for each thread
    DataType private_contribution = 0;
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      const auto& x0 = input0[row + col * input0_ldim];
      const auto& x1 = input1[row + col * input1_ldim];
      private_contribution += (x0 - mean0) * (x1 - mean1);
    }

    // Shared memory reduction to get contribution for each block
    /// @todo unroll loops
    __shared__ DataType shared_contribution[block_size];
    shared_contribution[tid] = private_contribution;
    for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_contribution[tid] += shared_contribution[tid + stride];
      }
    }
    if (tid == 0) {
      cuda::atomic_add(&contribution[col],
                       scale * shared_contribution[0]);
    }

  }

}

/** Compute gradients w.r.t. inputs. */
__global__
void covariance_backprop_kernel(El::Int height,
                                El::Int width,
                                DataType scale,
                                const DataType* __restrict__ gradient_wrt_output,
                                const DataType* __restrict__ input0,
                                El::Int input0_ldim,
                                const DataType* __restrict__ input1,
                                El::Int input1_ldim,
                                const DataType* __restrict__ means,
                                DataType* __restrict__ gradient_wrt_input0,
                                El::Int gradient_wrt_input0_ldim,
                                DataType* __restrict__ gradient_wrt_input1,
                                El::Int gradient_wrt_input1_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int nthreads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += nthreads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& dy = gradient_wrt_output[col];
    const auto& x0 = input0[row + col * input0_ldim];
    const auto& x1 = input1[row + col * input1_ldim];
    const auto& mean0 = means[2*col];
    const auto& mean1 = means[2*col+1];
    auto& dx0 = gradient_wrt_input0[row + col * gradient_wrt_input0_ldim];
    auto& dx1 = gradient_wrt_input1[row + col * gradient_wrt_input1_ldim];
    dx0 = dy * scale * (x1 - mean1);
    dx1 = dy * scale * (x0 - mean0);
  }
}

/** GPU forward prop implementation.
 *  We use a two-pass algorithm since it is more numerically stable
 *  than the naive single-pass algorithm.
 */
void fp_gpu(const AbsDistMat& input0,
            const AbsDistMat& input1,
            AbsDistMat& output,
            AbsDistMat& means,
            AbsDistMat& workspace,
            bool biased) {

  // Local matrices
  const auto& local_input0 = static_cast<const GPUMat&>(input0.LockedMatrix());
  const auto& local_input1 = static_cast<const GPUMat&>(input1.LockedMatrix());
  auto& local_means = static_cast<GPUMat&>(means.Matrix());
  auto& local_workspace = static_cast<GPUMat&>(workspace.Matrix());

  // Dimensions
  const auto& height = input0.Height();
  const auto& width = input0.Width();
  const auto& local_height = local_input0.Height();
  const auto& local_width = local_input0.Width();

  // Compute column-wise mean
  means.Empty(false);
  means.AlignWith(input0);
  El::Zeros(means, 2, width);
  if (!local_input0.IsEmpty()) {
    constexpr El::Int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    const auto& scale = DataType(1) / height;
    mean_contribution_kernel<block_size>
      <<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
        local_height, local_width, scale,
        local_input0.LockedBuffer(), local_input0.LDim(),
        local_input1.LockedBuffer(), local_input1.LDim(),
        local_means.Buffer());
  }
  El::AllReduce(means, means.RedundantComm());

  // Compute column-wise covariance
  workspace.Empty(false);
  workspace.AlignWith(input0);
  El::Zeros(workspace, 1, width);
  if (!local_input0.IsEmpty()) {
    constexpr El::Int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    const auto& scale = DataType(1) / (biased ? height : height - 1);
    covariance_contribution_kernel<block_size>
      <<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
        local_height, local_width, scale,
        local_input0.LockedBuffer(), local_input0.LDim(),
        local_input1.LockedBuffer(), local_input1.LDim(),
        local_means.LockedBuffer(),
        local_workspace.Buffer());
  }
  El::AllReduce(workspace, workspace.RedundantComm());
  El::Copy(workspace, output);

}

/** GPU backprop implementation.
 *  Means have already been computed in forward prop.
 */
void bp_gpu(const AbsDistMat& input0,
            const AbsDistMat& input1,
            const AbsDistMat& gradient_wrt_output,
            AbsDistMat& gradient_wrt_input0,
            AbsDistMat& gradient_wrt_input1,
            const AbsDistMat& means,
            AbsDistMat& workspace,
            bool biased) {

  // Local matrices
  const auto& local_input0 = static_cast<const GPUMat&>(input0.LockedMatrix());
  const auto& local_input1 = static_cast<const GPUMat&>(input1.LockedMatrix());
  auto& local_gradient_wrt_input0 = static_cast<GPUMat&>(gradient_wrt_input0.Matrix());
  auto& local_gradient_wrt_input1 = static_cast<GPUMat&>(gradient_wrt_input1.Matrix());
  const auto& local_means = static_cast<const GPUMat&>(means.LockedMatrix());
  auto& local_workspace = static_cast<GPUMat&>(workspace.Matrix());

  // Dimensions
  const auto& height = input0.Height();
  const auto& local_height = local_input0.Height();
  const auto& local_width = local_input0.Width();

  // Initialize workspace with gradients w.r.t. output
  El::Copy(gradient_wrt_output, workspace);

  // Compute gradients w.r.t. input
  const DataType scale = DataType(1) / (biased ? height : height - 1);
  constexpr El::Int block_size = 256;
  El::Int grid_size = (local_height * local_width + block_size - 1) / block_size;
  if (grid_size > 0) {
    covariance_backprop_kernel
      <<<grid_size, block_size, 0, El::GPUManager::Stream()>>>(
        local_height, local_width, scale,
        local_workspace.LockedBuffer(),
        local_input0.LockedBuffer(), local_input0.LDim(),
        local_input1.LockedBuffer(), local_input1.LDim(),
        local_means.LockedBuffer(),
        local_gradient_wrt_input0.Buffer(), local_gradient_wrt_input0.LDim(),
        local_gradient_wrt_input1.Buffer(), local_gradient_wrt_input1.LDim());
  }

}

} // namespace

template <>
void covariance_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(get_prev_activations(0),
         get_prev_activations(1),
         get_activations(),
         *m_means,
         *m_workspace,
         m_biased);
}

template <>
void covariance_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::bp_compute() {
  bp_gpu(get_prev_activations(0),
         get_prev_activations(1),
         get_prev_error_signals(),
         get_error_signals(0),
         get_error_signals(1),
         *m_means,
         *m_workspace,
         m_biased);
}

template <>
void covariance_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(get_prev_activations(0),
         get_prev_activations(1),
         get_activations(),
         *m_means,
         *m_workspace,
         m_biased);
}

template <>
void covariance_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::bp_compute() {
  bp_gpu(get_prev_activations(0),
         get_prev_activations(1),
         get_prev_error_signals(),
         get_error_signals(0),
         get_error_signals(1),
         *m_means,
         *m_workspace,
         m_biased);
}

} // namespace lbann
