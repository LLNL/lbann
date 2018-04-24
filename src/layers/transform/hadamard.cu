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

#include "lbann/layers/transform/hadamard.hpp"

namespace {

__global__ void ones_kernel(int height, int width,
                            lbann::DataType* x,
                            int x_ldim) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int pos = tid; pos < height * width; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    x[row + col * x_ldim] = lbann::DataType(1);
  }
}

__global__ void mult_kernel(int height, int width,
                            const lbann::DataType* __restrict__ x,
                            int x_ldim,
                            const lbann::DataType* __restrict__ y,
                            int y_ldim,
                            lbann::DataType* __restrict__ z,
                            int z_ldim) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int pos = tid; pos < height * width; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    z[row + col * z_ldim] = x[row + col * x_ldim] * y[row + col * y_ldim];
  }
}

__global__ void mult_assign_kernel(int height, int width,
                                   const lbann::DataType* __restrict__ x,
                                   int x_ldim,
                                   lbann::DataType* __restrict__ y,
                                   int y_ldim) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int pos = tid; pos < height * width; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    y[row + col * y_ldim] *= x[row + col * x_ldim];
  }
}

} // namespace

namespace lbann {
namespace hadamard_cuda {

void fp(cudnn::cudnn_manager& cudnn,
        int height,
        int width_per_gpu,
        const std::vector<std::vector<lbann::DataType*>>& inputs,
        const std::vector<int>& input_leading_dims,
        std::vector<lbann::DataType*>& output,
        int output_leading_dim) {

  // CUDA thread distribution
  const int size = height * width_per_gpu;
  const int block_dim = 256;
  const int grid_dim = (size + block_dim - 1) / block_dim;

  // Compute Hadamard product on each GPU
  const int num_gpus = cudnn.get_num_gpus();
  const int num_inputs = inputs.size();
  for (int i = 0; i < num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(cudnn.get_gpu(i)));
    switch (num_inputs) {
    case 0:
      ones_kernel<<<grid_dim, block_dim, 0, cudnn.get_stream(i)>>>(
        height, width_per_gpu, output[i], output_leading_dim);
      break;
    case 1:
      CHECK_CUDA(cudaMemcpy2DAsync(output[i],
                                   output_leading_dim * sizeof(lbann::DataType),
                                   inputs[0][i],
                                   input_leading_dims[0] * sizeof(lbann::DataType),
                                   height * sizeof(lbann::DataType),
                                   width_per_gpu,
                                   cudaMemcpyDeviceToDevice));
      break;
    default:
      mult_kernel<<<grid_dim, block_dim, 0, cudnn.get_stream(i)>>>(
        height, width_per_gpu,
        inputs[0][i], input_leading_dims[0],
        inputs[1][i], input_leading_dims[1],
        output[i], output_leading_dim);
      for (int j = 2; j < num_inputs; ++j) {
        mult_assign_kernel<<<grid_dim, block_dim, 0, cudnn.get_stream(i)>>>(
          height, width_per_gpu,
          inputs[j][i], input_leading_dims[j],
          output[i], output_leading_dim);
      }
    }
  }

}

void bp(cudnn::cudnn_manager& cudnn,
        int height,
        int width_per_gpu,
        const std::vector<std::vector<lbann::DataType*>>& inputs,
        const std::vector<int>& input_leading_dims,
        const std::vector<lbann::DataType*>& gradient_wrt_output,
        int gradient_wrt_output_leading_dim,
        std::vector<std::vector<lbann::DataType*>>& gradient_wrt_inputs,
        const std::vector<int>& gradient_wrt_input_leading_dims) {

  // CUDA thread distribution
  const int size = height * width_per_gpu;
  const int block_dim = 256;
  const int grid_dim = (size + block_dim - 1) / block_dim;

  // Compute gradients on each GPU
  const int num_gpus = cudnn.get_num_gpus();
  const int num_inputs = inputs.size();
  for (int i = 0; i < num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(cudnn.get_gpu(i)));
    switch (num_inputs) {
    case 0: break;
    case 1:
      CHECK_CUDA(cudaMemcpy2DAsync(gradient_wrt_inputs[0][i],
                                   gradient_wrt_input_leading_dims[0] * sizeof(lbann::DataType),
                                   gradient_wrt_output[i],
                                   gradient_wrt_output_leading_dim * sizeof(lbann::DataType),
                                   height * sizeof(lbann::DataType),
                                   width_per_gpu,
                                   cudaMemcpyDeviceToDevice));
      break;
    default:
      for (int j = 0; j < num_inputs; ++j) {
        int k = (j + 1) % num_inputs;
        mult_kernel<<<grid_dim, block_dim, 0, cudnn.get_stream(i)>>>(
          height, width_per_gpu,
          inputs[k][i], input_leading_dims[k],
          gradient_wrt_output[i], gradient_wrt_output_leading_dim,
          gradient_wrt_inputs[j][i], gradient_wrt_input_leading_dims[j]);
        for (k = (k+1) % num_inputs; k != j; k = (k+1) % num_inputs) {
          mult_assign_kernel<<<grid_dim, block_dim, 0, cudnn.get_stream(i)>>>(
            height, width_per_gpu,
            inputs[k][i], input_leading_dims[k],
            gradient_wrt_inputs[j][i], gradient_wrt_input_leading_dims[j]);
        }
      }
    }
  }

}
  
} // namespace hadamard_cuda
} // namespace lbann
