////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
//
// softmax_cuda.cu - GPU helper routines for softmax layer
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/activations/softmax.hpp"

namespace {

__global__ void fp_cutoff_kernel(int height, int width,
                                 lbann::DataType* output,
                                 int output_leading_dim,
                                 lbann::DataType cutoff) {
  const auto gid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto size = height * width;
  const auto num_threads = blockDim.x * gridDim.x;
  for (int pos = gid; pos < size; pos += num_threads) {
    const int row = pos % height;
    const int col = pos / height;
    auto& y = output[row + col * output_leading_dim];
    if (y < cutoff) { y = cutoff; }
  }
}

__global__ void bp_cutoff_kernel(int height, int width,
                                 const lbann::DataType* __restrict__ output,
                                 int output_leading_dim,
                                 lbann::DataType* __restrict__ gradient_wrt_input,
                                 int gradient_wrt_input_leading_dim,
                                 lbann::DataType cutoff) {
  const auto gid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto size = height * width;
  const auto num_threads = blockDim.x * gridDim.x;
  for (int pos = gid; pos < size; pos += num_threads) {
    const int row = pos % height;
    const int col = pos / height;
    const auto& y = output[row + col * output_leading_dim];
    if (y < cutoff) {
      auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_leading_dim];
      dx = lbann::DataType(0);
    }
  }
}

__global__ void max_local_col_entry_kernel(
  int height, int width,
  const lbann::DataType * __restrict__ input,
  int input_ldim,
  lbann::DataType * __restrict__ workspace) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int col = tid; col < width; col += num_threads) {
    const int col_offset = col*input_ldim;
    lbann::DataType max_entry = input[col_offset];
    for (int row = 1; row < height; ++row) {
      max_entry = fmax(max_entry, input[row + col_offset]);
    }
    workspace[col] = max_entry;
  }
}

__global__ void exp_and_col_sum_kernel(
  int height, int width,
  const lbann::DataType * __restrict__ input,
  int input_ldim,
  lbann::DataType * __restrict__ output,
  int output_ldim,
  lbann::DataType * __restrict__ workspace) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int col = tid; col < width; col += num_threads) {
    const int input_col_offset = col*input_ldim;
    const int output_col_offset = col*output_ldim;
    // Shift by the pre-computed maximum value for the column.
    const lbann::DataType shift = workspace[col];
    lbann::DataType sum = lbann::DataType(0);
    for (int row = 0; row < height; ++row) {
      const lbann::DataType y = exp(input[row + input_col_offset] - shift);
      output[row + output_col_offset] = y;
      sum += y;
    }
    workspace[col] = sum;
  }
}

__global__ void div_by_col_sums_and_cutoff_kernel(
  int height, int width,
  lbann::DataType * __restrict__ output,
  int output_ldim,
  const lbann::DataType * __restrict__ workspace,
  const lbann::DataType cutoff) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int col = tid; col < width; col += num_threads) {
    const int col_offset = col*output_ldim;
    const lbann::DataType scale = lbann::DataType(1) / workspace[col];
    for (int row = 0; row < height; ++row) {
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
      output[row + col_offset] = fmax(scale*output[row + col_offset], cutoff);
#else
      output[row + col_offset] *= scale;
#endif
    }
  }
}

__global__ void grad_wrt_input_and_cutoff_kernel(
  int height, int width,
  const lbann::DataType * __restrict__ output,
  int output_ldim,
  const lbann::DataType * __restrict__ workspace,
  const lbann::DataType * __restrict__ grad_wrt_output,
  int grad_wrt_output_ldim,
  lbann::DataType * __restrict__ grad_wrt_input,
  int grad_wrt_input_ldim,
  const lbann::DataType cutoff) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int col = tid; col < width; col += num_threads) {
    const lbann::DataType y_dot_dy = workspace[col];
    const int output_col_offset = col * output_ldim;
    const int grad_wrt_output_offset = col * grad_wrt_output_ldim;
    const int grad_wrt_input_offset = col * grad_wrt_input_ldim;
    for (int row = 0; row < height; ++row) {
      const auto& y = output[row + output_col_offset];
      auto& dx = grad_wrt_input[row + grad_wrt_input_offset];
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
      if (y <= cutoff) {
        dx = lbann::DataType(0);
      }
      else
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF
      {
        const auto& dy = grad_wrt_output[row + grad_wrt_output_offset];
        dx = y * (dy - y_dot_dy);
      }
    }
  }
}

}  // anonymous namespace

namespace lbann {
namespace softmax_cuda {

void fp_cutoff(int height, int width,
               DataType* output,
               int output_leading_dim,
               DataType cutoff,
               cudaStream_t stream) {
  const int size = height * width;
  if (size == 0) return;
  const int block_dim = 256;
  const int grid_dim = (size + block_dim - 1) / block_dim;
  fp_cutoff_kernel<<<grid_dim, block_dim, 0, stream>>>(
    height, width, output, output_leading_dim, cutoff);
}

void bp_cutoff(int height, int width,
               const DataType* output,
               int output_leading_dim,
               DataType* gradient_wrt_input,
               int gradient_wrt_input_leading_dim,
               DataType cutoff,
               cudaStream_t stream) {
  const int size = height * width;
  if (size == 0) return;
  const int block_dim = 256;
  const int grid_dim = (size + block_dim - 1) / block_dim;
  bp_cutoff_kernel<<<grid_dim, block_dim, 0, stream>>>(
    height, width,
    output, output_leading_dim,
    gradient_wrt_input, gradient_wrt_input_leading_dim,
    cutoff);
}

void max_local_col_entry(int height, int width,
                         const DataType * __restrict__ input,
                         int input_ldim,
                         DataType * __restrict__ workspace,
                         cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return;
  }
  const int block_dim = 256;
  const int grid_dim = (width + block_dim - 1) / block_dim;
  max_local_col_entry_kernel<<<grid_dim, block_dim, 0, stream>>>(
    height, width, input, input_ldim, workspace);
}

void exp_and_col_sum(int height, int width,
                     const DataType * __restrict__ input,
                     int input_ldim,
                     DataType * __restrict__ output,
                     int output_ldim,
                     DataType * __restrict__ workspace,
                     cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return;
  }
  const int block_dim = 256;
  const int grid_dim = (width + block_dim - 1) / block_dim;
  exp_and_col_sum_kernel<<<grid_dim, block_dim, 0, stream>>>(
    height, width, input, input_ldim, output, output_ldim, workspace);
}

void div_by_col_sums_and_cutoff(int height, int width,
                                DataType * __restrict__ output,
                                int output_ldim,
                                const DataType * __restrict__ workspace,
                                const DataType cutoff,
                                cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return;
  }
  const int block_dim = 256;
  const int grid_dim = (width + block_dim - 1) / block_dim;
  div_by_col_sums_and_cutoff_kernel<<<grid_dim, block_dim, 0, stream>>>(
    height, width, output, output_ldim, workspace, cutoff);
}

void grad_wrt_input_and_cutoff(int height, int width,
                               const DataType * __restrict__ output,
                               int output_ldim,
                               const DataType * __restrict__ workspace,
                               const DataType * __restrict__ grad_wrt_output,
                               int grad_wrt_output_ldim,
                               DataType * __restrict__ grad_wrt_input,
                               int grad_wrt_input_ldim,
                               const DataType cutoff,
                               cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return;
  }
  const int block_dim = 256;
  const int grid_dim = (width + block_dim - 1) / block_dim;
  grad_wrt_input_and_cutoff_kernel<<<grid_dim, block_dim, 0, stream>>>(
    height, width, output, output_ldim, workspace, grad_wrt_output,
    grad_wrt_output_ldim, grad_wrt_input, grad_wrt_input_ldim, cutoff);
}

} // namespace softmax_cuda
} // namespace lbann
