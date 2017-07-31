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
// fully_connected_cuda.cu - GPU helper routines for fully connected layer
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/learning/fully_connected_cuda.hpp"
#include <cassert>

namespace lbann {
namespace fully_connected_cuda {

__global__ void row_sum_kernel(El::Int h, El::Int w,
                               const DataType *m,
                               DataType factor,
                               DataType *v,
                               bool accum) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > h) return;
  DataType sum = 0;
  for (int i = 0; i < w; ++i) {
    sum += m[tid + i * h];
  }
  sum *= factor;
  if (accum) sum += v[tid];
  v[tid] = sum;
}



void row_sum(cudnn::cudnn_manager &cudnn,
             std::vector<DataType*> matrices,
             El::Int h, El::Int w,
             DataType factor,
             Mat &dest, 
             const std::vector<DataType*> &work_column) {
  int num_gpus = cudnn.get_num_gpus();
  int block_dim = 256;
  int grid_dim = h / block_dim + ((h % block_dim) ? 1 : 0);
  for (int i = 0; i < num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(cudnn.get_gpu(i)));
    row_sum_kernel<<<grid_dim, block_dim>>>(h, w, matrices[i], factor, work_column[i], false);
  }
#if 0  
  // inter-device reduction. requires peer access
  for (int peer_offset = 1; peer_offset < num_gpus; peer_offset*=2) {
    for (int i = 0; i < num_gpus; i+=peer_offset*2) {
      int peer = i + peer_offset;
      if (peer < num_gpus) {
        CHECK_CUDA(cudaSetDevice(cudnn.get_gpu(i)));
        row_sum_kernel<<<grid_dim, block_dim>>>(h, 1, work_column[peer], DataType(1), work_column[i], true);
      }
    }
  }
  // copy the reduced vector to host
  CHECK_CUDA(cudaSetDevice(cudnn.get_gpu(0)));
  CHECK_CUDA(cudaMemcpy(dest.Buffer(0, 0), work_column[0], h * sizeof(DataType),
                        cudaMemcpyDeviceToHost));
#else
  {
    // Get matrix properties
    assert(dest.Height() == h);
    assert(dest.Width() == 1);
  }
  cudnn.reduce_from_gpus(dest, work_column);
#endif
}

__global__ void add_tensor_kernel(DataType factor,
                                  DataType *bias,
                                  El::Int bias_h, El::Int bias_w,
                                  DataType beta,
                                  DataType *tensor,
                                  El::Int tensor_h, El::Int tensor_w) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > tensor_h * tensor_w) return;
  int h_idx = tid % tensor_h;
  int w_idx = tid / tensor_h;
  int bias_idx = (bias_h == 1) ? 0 : h_idx
      + ((bias_w == 1) ? 0 : w_idx) * bias_h;
  
  tensor[tid] = tensor[tid] * beta + factor * bias[bias_idx];
}


void add_tensor(DataType factor,
                DataType *bias,
                El::Int bias_h, El::Int bias_w,
                DataType beta,                
                DataType *tensor,
                El::Int tensor_h, El::Int tensor_w) {
  assert(bias_h == tensor_h || bias_h == 1);
  assert(bias_w == tensor_w || bias_w == 1);
  int block_dim = 256;
  int np = tensor_h * tensor_w;
  int grid_dim = np / block_dim + ((np % block_dim) ? 1 : 0);
  add_tensor_kernel<<<grid_dim, block_dim>>>(factor, bias, bias_h, bias_w,
                                             beta, tensor,
                                             tensor_h, tensor_w);
}


} // namespace fully_connected_cuda
} // namespace lbann
