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
#include "lbann/layers/activations/tanh.hpp"

namespace {

// Hyperbolic tangent function
__device__ inline float f(float x) { return tanhf(x); }
__device__ inline double f(double x) { return tanh(x); }
__device__ inline float df(float x) { 
  const float coshx = coshf(x);
  return 1 / (coshx * coshx);
}
__device__ inline double df(double x) { 
  const double coshx = cosh(x);
  return 1 / (coshx * coshx);
}

__global__ void fp_kernel(int height, int width,
                          const lbann::DataType* __restrict__ input,
                          int input_leading_dim,
                          lbann::DataType* __restrict__ output,
                          int output_leading_dim) {
  const int size = height * width;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int pos = tid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_leading_dim];
    output[row + col * output_leading_dim] = f(x);
  }    
}

__global__ void bp_kernel(int height, int width,
                          const lbann::DataType* __restrict__ input,
                          int input_leading_dim,
                          const lbann::DataType* __restrict__ gradient_wrt_output,
                          int gradient_wrt_output_leading_dim,
                          lbann::DataType* __restrict__ gradient_wrt_input,
                          int gradient_wrt_input_leading_dim) {
  const int size = height * width;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int pos = tid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_leading_dim];
    const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_leading_dim];
    const auto& dx = dy * df(x);
    gradient_wrt_input[row + col * gradient_wrt_input_leading_dim] = dx;
  }
}

}

namespace lbann {
namespace tanh_cuda {

void fp(cudnn::cudnn_manager& cudnn,
        int height,
        int width_per_gpu,
        const std::vector<lbann::DataType*>& input,
        int input_leading_dim,
        std::vector<lbann::DataType*>& output,
        int output_leading_dim) {

  // CUDA thread distribution
  const int size = height * width_per_gpu;
  const int block_dim = 256;
  const int grid_dim = (size + block_dim - 1) / block_dim;

  // Launch kernel on each GPU
  const int num_gpus = cudnn.get_num_gpus();
  for (int i = 0; i < num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(cudnn.get_gpu(i)));
    fp_kernel<<<grid_dim, block_dim, 0, cudnn.get_stream(i)>>>(
      height, width_per_gpu,
      input[i], input_leading_dim,
      output[i], output_leading_dim);
  }

}

void bp(cudnn::cudnn_manager& cudnn,
        int height,
        int width_per_gpu,
        const std::vector<lbann::DataType*>& input,
        int input_leading_dim,
        const std::vector<lbann::DataType*>& gradient_wrt_output,
        int gradient_wrt_output_leading_dim,
        std::vector<lbann::DataType*>& gradient_wrt_input,
        int gradient_wrt_input_leading_dim) {

  // CUDA thread distribution
  const int size = height * width_per_gpu;
  const int block_dim = 256;
  const int grid_dim = (size + block_dim - 1) / block_dim;

  // Launch kernel on each GPU
  const int num_gpus = cudnn.get_num_gpus();
  for (int i = 0; i < num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(cudnn.get_gpu(i)));
    bp_kernel<<<grid_dim, block_dim, 0, cudnn.get_stream(i)>>>(
      height, width_per_gpu,
      input[i], input_leading_dim,
      gradient_wrt_output[i], gradient_wrt_output_leading_dim,
      gradient_wrt_input[i], gradient_wrt_input_leading_dim);
  }
}

} // namespace tanh_cuda
} // namespace lbann
