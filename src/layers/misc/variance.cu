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

#include "lbann/layers/misc/variance.hpp"

namespace lbann {

namespace {

template <El::Int block_size>
__global__ void variance_contribution_kernel(El::Int height,
                                             El::Int width,
                                             DataType scale,
                                             const DataType* __restrict__ input,
                                             El::Int input_ldim,
                                             const DataType* __restrict__ means,
                                             DataType* __restrict__ contribution) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;

  // Compute local contribution for each matrix column
  for (El::Int col = bidy; col < width; col += gridDim.y) {
    const auto& mean = means[col];

    // Compute contributions for each thread
    DataType private_contribution = 0;
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      const auto& diff = input[row + col * input_ldim] - mean;
      private_contribution += diff * diff;
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

__global__
void variance_backprop_kernel(El::Int height,
                              El::Int width,
                              DataType scale,
                              const DataType* __restrict__ gradient_wrt_output,
                              const DataType* __restrict__ input,
                              El::Int input_ldim,
                              const DataType* __restrict__ means,
                              DataType* __restrict__ gradient_wrt_input,
                              El::Int gradient_wrt_input_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int nthreads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += nthreads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& dy = gradient_wrt_output[col];
    const auto& x = input[row + col * input_ldim];
    const auto& mean = means[col];
    auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
    dx = dy * scale * (x - mean);
  }
}

/** GPU forward prop implementation.
 *  We use a two-pass algorithm since it is more numerically stable
 *  than the naive single-pass algorithm.
 */
void fp_gpu(const AbsDistMat& input,
            AbsDistMat& output,
            AbsDistMat& means,
            AbsDistMat& workspace,
            bool biased) {

  // Local matrices
  const auto& local_input = static_cast<const GPUMat&>(input.LockedMatrix());
  auto& local_means = static_cast<GPUMat&>(means.Matrix());
  auto& local_workspace = static_cast<GPUMat&>(workspace.Matrix());

  // Dimensions
  const auto& height = input.Height();
  const auto& width = input.Width();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // Compute column-wise mean
  means.Empty(false);
  means.AlignWith(input);
  means.Resize(1, width);
  GPUMat ones;
#ifdef HYDROGEN_HAVE_CUB
  ones.SetMemoryMode(1); // Use CUB GPU memory pool
#endif // HYDROGEN_HAVE_CUB
  ones.Resize(local_height, 1);
  El::Fill(ones, DataType(1));
  El::Gemv(El::TRANSPOSE,
           DataType(1) / height, local_input, ones,
           DataType(0), local_means);
  El::AllReduce(means, means.RedundantComm());

  // Compute column-wise variance
  workspace.Empty(false);
  workspace.AlignWith(input);
  El::Zeros(workspace, 1, width);
  if (!local_input.IsEmpty()) {
    constexpr El::Int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    const auto& scale = DataType(1) / (biased ? height : height - 1);
    variance_contribution_kernel<block_size>
      <<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
        local_height, local_width, scale,
        local_input.LockedBuffer(), local_input.LDim(),
        local_means.LockedBuffer(),
        local_workspace.Buffer());
  }
  El::AllReduce(workspace, workspace.RedundantComm());
  El::Copy(workspace, output);

}

/** GPU backprop implementation.
 *  Means have already been computed in forward prop.
 */
void bp_gpu(const AbsDistMat& input,
            const AbsDistMat& gradient_wrt_output,
            AbsDistMat& gradient_wrt_input,
            const AbsDistMat& means,
            AbsDistMat& workspace,
            bool biased) {

  // Local matrices
  const auto& local_input = static_cast<const GPUMat&>(input.LockedMatrix());
  auto& local_gradient_wrt_input = static_cast<GPUMat&>(gradient_wrt_input.Matrix());
  const auto& local_means = static_cast<const GPUMat&>(means.LockedMatrix());
  auto& local_workspace = static_cast<GPUMat&>(workspace.Matrix());

  // Dimensions
  const auto& height = input.Height();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // Initialize workspace with gradients w.r.t. output
  El::Copy(gradient_wrt_output, workspace);

  // Compute gradients w.r.t. input
  const DataType scale = DataType(2) / (biased ? height : height - 1);
  constexpr El::Int block_size = 256;
  El::Int grid_size = (local_height * local_width + block_size - 1) / block_size;
  if (grid_size > 0) {
    variance_backprop_kernel
      <<<grid_size, block_size, 0, El::GPUManager::Stream()>>>(
        local_height, local_width, scale,
        local_workspace.LockedBuffer(),
        local_input.LockedBuffer(), local_input.LDim(),
        local_means.LockedBuffer(),
        local_gradient_wrt_input.Buffer(), local_gradient_wrt_input.LDim());
  }

}

} // namespace

template <>
void variance_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(get_prev_activations(),
         get_activations(),
         *m_means,
         *m_workspace,
         m_biased);
}

template <>
void variance_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::bp_compute() {
  bp_gpu(get_prev_activations(),
         get_prev_error_signals(),
         get_error_signals(),
         *m_means,
         *m_workspace,
         m_biased);
}

template <>
void variance_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(get_prev_activations(),
         get_activations(),
         *m_means,
         *m_workspace,
         m_biased);
}

template <>
void variance_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::bp_compute() {
  bp_gpu(get_prev_activations(),
         get_prev_error_signals(),
         get_error_signals(),
         *m_means,
         *m_workspace,
         m_biased);
}

} // namespace lbann
