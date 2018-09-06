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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/activations/relu.hpp"
#include "lbann/utils/cuda.hpp"

namespace lbann {
namespace {

__global__ void fp_kernel(int height, int width,
                          const DataType* __restrict__ input,
                          int input_leading_dim,
                          DataType* __restrict__ output,
                          int output_leading_dim) {
  const auto& gid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& size = height * width;
  const auto& num_threads = blockDim.x * gridDim.x;
  for (int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_leading_dim];
    auto& y = output[row + col * output_leading_dim];
    y = cuda::max(x, DataType(0));
  }
}

__global__ void bp_kernel(int height, int width,
                          const DataType* __restrict__ input,
                          int input_leading_dim,
                          const DataType* __restrict__ gradient_wrt_output,
                          int gradient_wrt_output_leading_dim,
                          DataType* __restrict__ gradient_wrt_input,
                          int gradient_wrt_input_leading_dim) {
  const auto& gid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& size = height * width;
  const auto& num_threads = blockDim.x * gridDim.x;
  for (int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_leading_dim];
    const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_leading_dim];
    auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_leading_dim];
    if (x > DataType(0)) {
      dx = dy;
    } else {
      dx = DataType(0);
    }
  }
}

void fp(const AbsMat& input, AbsMat& output) {
  const auto& height = input.Height();
  const auto& width = input.Width();
  const auto& block_dim = 256;
  const auto& grid_dim = (height * width + block_dim - 1) / block_dim;
  if (grid_dim > 0) {
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    fp_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
      height, width,
      input.LockedBuffer(), input.LDim(),
      output.Buffer(), output.LDim());
  }
}

void bp(const AbsMat& input,
        const AbsMat& gradient_wrt_output,
        AbsMat& gradient_wrt_input) {
  const auto& height = input.Height();
  const auto& width = input.Width();
  const auto& block_dim = 256;
  const auto& grid_dim = (height * width + block_dim - 1) / block_dim;
  if (grid_dim > 0) {
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    bp_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
      height, width,
      input.LockedBuffer(), input.LDim(),
      gradient_wrt_output.LockedBuffer(), gradient_wrt_output.LDim(),
      gradient_wrt_input.Buffer(), gradient_wrt_input.LDim());
  }
}

} // namespace

template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {
  fp(get_local_prev_activations(), get_local_activations());
}
template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
  bp(get_local_prev_activations(),
     get_local_prev_error_signals(),
     get_local_error_signals());
}
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
  fp(get_local_prev_activations(), get_local_activations());
}
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
  bp(get_local_prev_activations(),
     get_local_prev_error_signals(),
     get_local_error_signals());
}
  
} // namespace lbann
