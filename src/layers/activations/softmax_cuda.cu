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

#include "lbann/layers/activations/softmax_cuda.hpp"

#define LBANN_ENABLE_SOFTMAX_CUTOFF_CUDA

namespace lbann {
namespace softmax_cuda {

__global__ void fp_cutoff_kernel(DataType *activations,
                                 El::Int num_elms,
                                 DataType min_output) {
  El::Int tid = ((El::Int)blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid > num_elms) return;
  DataType x = activations[tid];
  x = x > min_output ? x : min_output;
  activations[tid] = x;
}

void fp_cutoff(cudnn::cudnn_manager &cudnn,
               const std::vector<DataType*> &activations,
               El::Int h, El::Int w,
               DataType min_output) {
#ifndef LBANN_ENABLE_SOFTMAX_CUTOFF_CUDA
  return;
#else
  El::Int num_elms = h * w;  
  int num_gpus = cudnn.get_num_gpus();
  int block_dim = 256;
  int grid_dim = num_gpus / block_dim + ((num_gpus % block_dim) ? 1 : 0);
  for (int i = 0; i < num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(cudnn.get_gpu(i)));
    fp_cutoff_kernel<<<grid_dim, block_dim>>>(activations[i], num_elms,
                                              min_output);
  }
#endif
}

__global__ void bp_cutoff_kernel(const DataType *activations,
                                 DataType *error_signals,
                                 El::Int num_elms,
                                 DataType min_output) {
  El::Int tid = ((El::Int)blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid > num_elms) return;
  DataType a = activations[tid];
  DataType e = error_signals[tid];  
  e = a > min_output ? e : DataType(0);
  error_signals[tid] = e;
}

void bp_cutoff(cudnn::cudnn_manager &cudnn,
               const std::vector<DataType*> &activations,
               const std::vector<DataType*> &error_signals,               
               El::Int h, El::Int w,
               DataType min_output) {
#ifndef LBANN_ENABLE_SOFTMAX_CUTOFF_CUDA
  return;
#else
  El::Int num_elms = h * w;  
  int num_gpus = cudnn.get_num_gpus();
  int block_dim = 256;
  int grid_dim = num_gpus / block_dim + ((num_gpus % block_dim) ? 1 : 0);
  for (int i = 0; i < num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(cudnn.get_gpu(i)));
    bp_cutoff_kernel<<<grid_dim, block_dim>>>(activations[i], error_signals[i],
                                              num_elms, min_output);
  }
#endif
}


__global__ void bp_compute_cross_entropy_shortcut_kernel(const DataType *activations,
                                                         const DataType *prev_error_signals,
                                                         DataType *error_signals,
                                                         El::Int num_elms,
                                                         DataType min_output) {
  El::Int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > num_elms) return;
  DataType a = activations[tid];
  DataType p = prev_error_signals[tid];
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF_CUDA
  DataType e = a > min_output ? a - p : DataType(0);
#else
  DataType e = a - p;
#endif
  error_signals[tid] = e;
  return;
}

void bp_compute_cross_entropy_shortcut(cudnn::cudnn_manager &cudnn,
                                       const std::vector<DataType*> &activations,
                                       const std::vector<DataType*> &prev_error_signals,
                                       const std::vector<DataType*> &error_signals,
                                       El::Int h, El::Int w,
                                       DataType min_output) {
  El::Int num_elms = h * w;                                       
  int block_dim = 256;
  int grid_dim = num_elms / block_dim + ((num_elms % block_dim) ? 1 : 0);
  int num_gpus = cudnn.get_num_gpus();  
  for (int i = 0; i < num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(cudnn.get_gpu(i)));
    bp_compute_cross_entropy_shortcut_kernel<<<
      grid_dim, block_dim, 0, cudnn.get_stream(i)>>>(
          activations[i], prev_error_signals[i], error_signals[i], num_elms, min_output);
  }
}

} // namespace softmax_cuda
} // namespace lbann
