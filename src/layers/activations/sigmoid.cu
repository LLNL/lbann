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

namespace {

// Sigmoid function
#if __CUDA_ARCH__ >= 530
__device__ inline __half sigmoid(__half x) {
  return __hdiv(__float2half(1.f),
                __hadd(__float2half(1.f), hexp(__hneg(x))));
}
#endif // __CUDA_ARCH__ >= 530
__device__ inline float sigmoid(float x) {
  return 1 / (1.0f + expf(-x));
}
__device__ inline double sigmoid(double x) {
  return 1 / (1.0 + exp(-x));
}


__global__ void fp_kernel(int height, int width,
                          const lbann::DataType* __restrict__ input,
                          int input_leading_dim,
                          lbann::DataType* __restrict__ output,
                          int output_leading_dim,
                          lbann::DataType cutoff) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto size = height * width;
  const auto num_threads = blockDim.x * gridDim.x;
  for (int pos = tid; pos < size; pos += num_threads) {

    // Get input value
    const int row = tid % height;
    const int col = tid / height;
    auto x = input[row + col * input_leading_dim];

    // Compute output value
  #ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (x < -cutoff) { x = -cutoff; }
    if (x > cutoff) { x = cutoff; }
  #endif // LBANN_ENABLE_SIGMOID_CUTOFF
    output[row + col * output_leading_dim] = sigmoid(x);

  }
}

__global__ void bp_kernel(int height, int width,
                          const lbann::DataType* __restrict__ input,
                          int input_leading_dim,
                          const lbann::DataType* __restrict__ gradient_wrt_output,
                          int gradient_wrt_output_leading_dim,
                          lbann::DataType* __restrict__ gradient_wrt_input,
                          int gradient_wrt_input_leading_dim,
                          lbann::DataType cutoff) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto size = height * width;
  const auto num_threads = blockDim.x * gridDim.x;
  for (int pos = tid; pos < size; pos += num_threads) {

    // Get input value
    const int row = tid % height;
    const int col = tid / height;
    const auto x = input[row + col * input_leading_dim];
    lbann::DataType dx = lbann::DataType(0);
  #ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (-cutoff <= x && x <= cutoff)
  #endif // LBANN_ENABLE_SIGMOID_CUTOFF
    {
      const auto& dy
        = gradient_wrt_output[row + col * gradient_wrt_output_leading_dim];
      const auto& sigx = sigmoid(x);
      dx = dy * sigx * (lbann::DataType(1) - sigx);
    }
    gradient_wrt_input[row + col * gradient_wrt_input_leading_dim] = dx;

  }
}

}

namespace lbann {
namespace sigmoid_cuda {

void fp(cudnn::cudnn_manager& cudnn,
        int height,
        int width,
        const lbann::DataType* input,
        int input_leading_dim,
        lbann::DataType* output,
        int output_leading_dim,
        lbann::DataType cutoff) {
  const int block_dim = 256;
  const int grid_dim = (height * width + block_dim - 1) / block_dim;
  CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
  fp_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
    height, width,
    input, input_leading_dim,
    output, output_leading_dim,
    cutoff);
}

void bp(cudnn::cudnn_manager& cudnn,
        int height,
        int width,
        const lbann::DataType* input,
        int input_leading_dim,
        const lbann::DataType* gradient_wrt_output,
        int gradient_wrt_output_leading_dim,
        lbann::DataType* gradient_wrt_input,
        int gradient_wrt_input_leading_dim,
        lbann::DataType cutoff) {
  const int block_dim = 256;
  const int grid_dim = (height * width + block_dim - 1) / block_dim;
  CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
  bp_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
    height, width,
    input, input_leading_dim,
    gradient_wrt_output, gradient_wrt_output_leading_dim,
    gradient_wrt_input, gradient_wrt_input_leading_dim,
    cutoff);
}

} // namespace sigmoid_cuda
} // namespace lbann
