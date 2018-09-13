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

#include "math.h"
#include "lbann/layers/activations/sigmoid.hpp"

namespace lbann {
namespace {

// Sigmoid function
#if __CUDA_ARCH__ >= 530
__device__ inline __half sigmoid(__half x) {
  static_cast<void>(static_cast<__half (*)(__half)>(sigmoid)); // Suppress "unused function" warning
  return __hdiv(__float2half(1.f),
                __hadd(__float2half(1.f), hexp(__hneg(x))));
}
#endif // __CUDA_ARCH__ >= 530
__device__ inline float sigmoid(float x) {
  static_cast<void>(static_cast<float (*)(float)>(sigmoid)); // Suppress "unused function" warning
  return 1 / (1.0f + expf(-x));
}
__device__ inline double sigmoid(double x) {
  static_cast<void>(static_cast<double (*)(double)>(sigmoid)); // Suppress "unused function" warning
  return 1 / (1.0 + exp(-x));
}
  
__global__ void fp_kernel(El::Int height, El::Int width,
                          const DataType* __restrict__ input,
                          El::Int input_leading_dim,
                          DataType* __restrict__ output,
                          El::Int output_leading_dim,
                          DataType eps) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_leading_dim];
    auto y = sigmoid(x);
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (y <= eps) { y = eps; }
    else if (y >= DataType(1) - eps) { y = DataType(1) - eps; }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    output[row + col * output_leading_dim] = y;
  }
}

__global__ void bp_kernel(El::Int height, El::Int width,
                          const DataType* __restrict__ input,
                          El::Int input_leading_dim,
                          const DataType* __restrict__ gradient_wrt_output,
                          El::Int gradient_wrt_output_leading_dim,
                          DataType* __restrict__ gradient_wrt_input,
                          El::Int gradient_wrt_input_leading_dim,
                          DataType eps) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_leading_dim];
    const auto& y = sigmoid(x);
    const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_leading_dim];
    auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_leading_dim];
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (y <= eps || y >= DataType(1) - eps) {
      dx = DataType(0);
      continue;
    }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    dx = dy * y * (DataType(1) - y);
  }
}

void fp(const AbsMat& input, AbsMat& output, DataType eps) {
  const auto& height = input.Height();
  const auto& width = input.Width();
  const auto& block_dim = 256;
  const auto& grid_dim = (height * width + block_dim - 1) / block_dim;
  if (grid_dim > 0) {
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    fp_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
      height, width,
      input.LockedBuffer(), input.LDim(),
      output.Buffer(), output.LDim(),
      eps);
  }
}

void bp(const AbsMat& input,
        const AbsMat& gradient_wrt_output,
        AbsMat& gradient_wrt_input,
        DataType eps) {
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
      gradient_wrt_input.Buffer(), gradient_wrt_input.LDim(),
      eps);
  }
}
  
} // namespace

template <>
void sigmoid_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
  fp(get_local_prev_activations(), get_local_activations(), eps);
}
template <>
void sigmoid_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
  bp(get_local_prev_activations(),
     get_local_prev_error_signals(),
     get_local_error_signals(),
     eps);
}
template <>
void sigmoid_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {
  fp(get_local_prev_activations(), get_local_activations(), eps);
}
template <>
void sigmoid_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
  bp(get_local_prev_activations(),
     get_local_prev_error_signals(),
     get_local_error_signals(),
     eps);
}

} // namespace lbann
