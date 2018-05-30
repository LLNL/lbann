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
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto size = height * width;
  const auto num_threads = blockDim.x * gridDim.x;
  for (int pos = tid; pos < size; pos += num_threads) {
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
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto size = height * width;
  const auto num_threads = blockDim.x * gridDim.x;
  for (int pos = tid; pos < size; pos += num_threads) {
    const int row = pos % height;
    const int col = pos / height;
    const auto& y = output[row + col * output_leading_dim];
    if (y < cutoff) { 
      auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_leading_dim];
      dx = lbann::DataType(0);
    }
  }
}
  
}

namespace lbann {
namespace softmax_cuda {

void fp_cutoff(cudnn::cudnn_manager& cudnn,
               int height, int width,
               DataType* output,
               int output_leading_dim,
               DataType cutoff) {
  const int size = height * width;
  const int block_dim = 256;
  const int grid_dim = (size + block_dim - 1) / block_dim;
  CHECK_CUDA(cudaSetDevice(cudnn.get_gpu()));
  fp_cutoff_kernel<<<grid_dim, block_dim, 0, cudnn.get_stream()>>>(
    height, width, output, output_leading_dim, cutoff);
}

void bp_cutoff(cudnn::cudnn_manager& cudnn,
               int height, int width,
               const DataType* output,
               int output_leading_dim,
               DataType* gradient_wrt_input,
               int gradient_wrt_input_leading_dim,
               DataType cutoff) {
  const int size = height * width;
  const int block_dim = 256;
  const int grid_dim = (size + block_dim - 1) / block_dim;
  CHECK_CUDA(cudaSetDevice(cudnn.get_gpu()));
  bp_cutoff_kernel<<<grid_dim, block_dim, 0, cudnn.get_stream()>>>(
    height, width,
    output, output_leading_dim,
    gradient_wrt_input, gradient_wrt_input_leading_dim,
    cutoff);
}

} // namespace softmax_cuda
} // namespace lbann
