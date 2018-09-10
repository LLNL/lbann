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

#include "lbann/layers/activations/logsoftmax.hpp"

namespace {

__global__ void max_local_col_entry_kernel(
  int height, int width,
  const lbann::DataType * __restrict__ input,
  int input_ldim,
  lbann::DataType * __restrict__ shift_workspace) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int col = tid; col < width; col += num_threads) {
    const int col_offset = col*input_ldim;
    lbann::DataType max_entry = input[col_offset];
    for (int row = 1; row < height; ++row) {
      max_entry = fmax(max_entry, input[row + col_offset]);
    }
    shift_workspace[col] = max_entry;
  }
}

__global__ void exp_and_col_sum_kernel(
  int height, int width,
  const lbann::DataType * __restrict__ input,
  int input_ldim,
  lbann::DataType * __restrict__ shift_workspace,
  lbann::DataType * __restrict__ sum_workspace) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int col = tid; col < width; col += num_threads) {
    const int input_col_offset = col*input_ldim;
    // Shift by the pre-computed maximum value for the column.
    const lbann::DataType shift = shift_workspace[col];
    lbann::DataType sum = lbann::DataType(0);
    for (int row = 0; row < height; ++row) {
      const lbann::DataType y = exp(input[row + input_col_offset] - shift);
      sum += y;
    }
    sum_workspace[col] = sum;
  }
}

__global__ void sub_by_col_sums_and_shift_kernel(
  int height, int width,
  const lbann::DataType * __restrict__ input,
  int input_ldim,
  lbann::DataType * __restrict__ output,
  int output_ldim,
  const lbann::DataType * __restrict__ shift_workspace,
  const lbann::DataType * __restrict__ sum_workspace) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int col = tid; col < width; col += num_threads) {
    const int input_col_offset = col*input_ldim;
    const int output_col_offset = col*output_ldim;
    const lbann::DataType shift = shift_workspace[col];
    const lbann::DataType lse = log(sum_workspace[col]);
    for (int row = 0; row < height; ++row) {
      output[row + output_col_offset] = input[row + input_col_offset] - lse - shift;
    }
  }
}

__global__ void grad_wrt_input_kernel(
  int height, int width,
  const lbann::DataType * __restrict__ output,
  int output_ldim,
  const lbann::DataType * __restrict__ sum_workspace,
  const lbann::DataType * __restrict__ grad_wrt_output,
  int grad_wrt_output_ldim,
  lbann::DataType * __restrict__ grad_wrt_input,
  int grad_wrt_input_ldim) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int col = tid; col < width; col += num_threads) {
    const lbann::DataType sum = sum_workspace[col];
    const int output_col_offset = col * output_ldim;
    const int grad_wrt_output_offset = col * grad_wrt_output_ldim;
    const int grad_wrt_input_offset = col * grad_wrt_input_ldim;
    for (int row = 0; row < height; ++row) {
      const auto& y = output[row + output_col_offset];
      auto& dx = grad_wrt_input[row + grad_wrt_input_offset];
      {
        const auto& dy = grad_wrt_output[row + grad_wrt_output_offset];
        dx = dy - exp(y) * sum;
      }
    }
  }
}

}  // anonymous namespace

namespace lbann {
namespace logsoftmax_cuda {

void max_local_col_entry(int height, int width,
                         const DataType * __restrict__ input,
                         int input_ldim,
                         DataType * __restrict__ shift_workspace,
                         cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return;
  }
  const int block_dim = 256;
  const int grid_dim = (width + block_dim - 1) / block_dim;
  max_local_col_entry_kernel<<<grid_dim, block_dim, 0, stream>>>(
    height, width, input, input_ldim, shift_workspace);
}

void exp_and_col_sum(int height, int width,
                     const DataType * __restrict__ input,
                     int input_ldim,
                     DataType * __restrict__ shift_workspace,
                     DataType * __restrict__ sum_workspace,
                     cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return;
  }
  const int block_dim = 256;
  const int grid_dim = (width + block_dim - 1) / block_dim;
  exp_and_col_sum_kernel<<<grid_dim, block_dim, 0, stream>>>(
    height, width, input, input_ldim, shift_workspace, sum_workspace);
}

void sub_by_col_sums_and_shift(int height, int width,
                                const DataType * __restrict__ input,
                                int input_ldim,
                                DataType * __restrict__ output,
                                int output_ldim,
                                const DataType * __restrict__ shift_workspace,
                                const DataType * __restrict__ sum_workspace,
                                cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return;
  }
  const int block_dim = 256;
  const int grid_dim = (width + block_dim - 1) / block_dim;
  sub_by_col_sums_and_shift_kernel<<<grid_dim, block_dim, 0, stream>>>(
    height, width, input, input_ldim, output, output_ldim, shift_workspace, sum_workspace);
}

void grad_wrt_input(int height, int width,
                               const DataType * __restrict__ output,
                               int output_ldim,
                               const DataType * __restrict__ sum_workspace,
                               const DataType * __restrict__ grad_wrt_output,
                               int grad_wrt_output_ldim,
                               DataType * __restrict__ grad_wrt_input,
                               int grad_wrt_input_ldim,
                               cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return;
  }
  const int block_dim = 256;
  const int grid_dim = (width + block_dim - 1) / block_dim;
  grad_wrt_input_kernel<<<grid_dim, block_dim, 0, stream>>>(
    height, width, output, output_ldim, sum_workspace, grad_wrt_output,
    grad_wrt_output_ldim, grad_wrt_input, grad_wrt_input_ldim);
}

} // namespace logsoftmax_cuda
} // namespace lbann
