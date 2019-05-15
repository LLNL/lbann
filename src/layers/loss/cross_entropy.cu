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

#include "lbann/layers/loss/cross_entropy.hpp"
#include "lbann/utils/exception.hpp"
#include "math.h"

namespace lbann {

namespace {

template <int block_size>
__global__ void fp_kernel(int height, int width,
                          const DataType* __restrict__ prediction,
                          int prediction_ldim,
                          const DataType* __restrict__ ground_truth,
                          int ground_truth_ldim,
                          DataType* __restrict__ contribution) {

  // Indices
  const int tid = threadIdx.x;
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;
  const int nthreadsx = blockDim.x * gridDim.x;

  // Compute local contribution for each matrix column
  for (int col = bidy; col < width; col += gridDim.y) {

    // Compute contributions for each thread
    DataType private_contribution = DataType(0);
    for (int row = gidx; row < height; row += nthreadsx) {
      const auto& xhat = ground_truth[row + col * ground_truth_ldim];
      if (xhat > DataType(0)) {
        const auto& x = prediction[row + col * prediction_ldim];
        private_contribution += - xhat * log(x);
      }
    }

    // Shared memory reduction to get contribution for each block
    /// @todo unroll loops
    __shared__ DataType shared_contribution[block_size];
    shared_contribution[tid] = private_contribution;
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_contribution[tid] += shared_contribution[tid + stride];
      }
    }
    if (tid == 0) {
      cuda::atomic_add(&contribution[col], shared_contribution[0]);
    }

  }

}

void local_fp_gpu(const AbsMat& local_prediction,
                  const AbsMat& local_ground_truth,
                  AbsMat& local_contribution) {
  El::Zero(local_contribution);
  const auto& height = local_prediction.Height();
  const auto& width = local_prediction.Width();
  if (height > 0 && width > 0) {
    const int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (height + block_size - 1) / block_size;
    grid_dims.y = width;
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    fp_kernel<block_size>
      <<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
        height, width,
        local_prediction.LockedBuffer(), local_prediction.LDim(),
        local_ground_truth.LockedBuffer(), local_ground_truth.LDim(),
        local_contribution.Buffer());
  }
}

template <int block_size>
__global__ void bp_kernel(int height, int width,
                          const DataType* __restrict__ prediction,
                          int prediction_ldim,
                          const DataType* __restrict__ ground_truth,
                          int ground_truth_ldim,
                          const DataType* __restrict__ gradient_wrt_output,
                          DataType* __restrict__ gradient_wrt_prediction,
                          int gradient_wrt_prediction_ldim,
                          DataType* __restrict__ gradient_wrt_ground_truth,
                          int gradient_wrt_ground_truth_ldim) {

  // Indices
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;
  const int nthreadsx = blockDim.x * gridDim.x;

  // Compute gradients
  for (int col = bidy; col < width; col += gridDim.y) {
    const auto& dy = gradient_wrt_output[col];
    for (int row = gidx; row < height; row += nthreadsx) {
      const auto& x = prediction[row + col * prediction_ldim];
      const auto& xhat = ground_truth[row + col * ground_truth_ldim];
      auto& dx = gradient_wrt_prediction[row + col * gradient_wrt_prediction_ldim];
      auto& dxhat = gradient_wrt_ground_truth[row + col * gradient_wrt_ground_truth_ldim];
      dx = (xhat > DataType(0)) ? - dy * xhat / x : DataType(0);
      dxhat = - dy * std::log(x);
    }
  }

}

void local_bp_gpu(const AbsMat& local_prediction,
                  const AbsMat& local_ground_truth,
                  const AbsMat& local_gradient_wrt_output,
                  AbsMat& local_gradient_wrt_prediction,
                  AbsMat& local_gradient_wrt_ground_truth) {
  const auto& height = local_prediction.Height();
  const auto& width = local_prediction.Width();
  if (height > 0 && width > 0) {
    const int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (height + block_size - 1) / block_size;
    grid_dims.y = width;
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    bp_kernel<block_size>
      <<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
        height, width,
        local_prediction.LockedBuffer(), local_prediction.LDim(),
        local_ground_truth.LockedBuffer(), local_ground_truth.LDim(),
        local_gradient_wrt_output.LockedBuffer(),
        local_gradient_wrt_prediction.Buffer(),
        local_gradient_wrt_prediction.LDim(),
        local_gradient_wrt_ground_truth.Buffer(),
        local_gradient_wrt_ground_truth.LDim());
  }
}

} // namespace

template <>
void cross_entropy_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::local_fp_compute(const AbsMat& local_prediction,
                        const AbsMat& local_ground_truth,
                        AbsMat& local_contribution) {
  local_fp_gpu(local_prediction, local_ground_truth, local_contribution);
}

template <>
void cross_entropy_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::local_bp_compute(const AbsMat& local_prediction,
                        const AbsMat& local_ground_truth,
                        const AbsMat& local_gradient_wrt_output,
                        AbsMat& local_gradient_wrt_prediction,
                        AbsMat& local_gradient_wrt_ground_truth) {
  local_bp_gpu(local_prediction,
               local_ground_truth,
               local_gradient_wrt_output,
               local_gradient_wrt_prediction,
               local_gradient_wrt_ground_truth);
}

template <>
void cross_entropy_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::local_fp_compute(const AbsMat& local_prediction,
                        const AbsMat& local_ground_truth,
                        AbsMat& local_contribution) {
  local_fp_gpu(local_prediction, local_ground_truth, local_contribution);
}

template <>
void cross_entropy_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::local_bp_compute(const AbsMat& local_prediction,
                        const AbsMat& local_ground_truth,
                        const AbsMat& local_gradient_wrt_output,
                        AbsMat& local_gradient_wrt_prediction,
                        AbsMat& local_gradient_wrt_ground_truth) {
  local_bp_gpu(local_prediction,
               local_ground_truth,
               local_gradient_wrt_output,
               local_gradient_wrt_prediction,
               local_gradient_wrt_ground_truth);
}

} // namespace lbann
