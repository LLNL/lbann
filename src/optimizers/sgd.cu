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

#include "lbann/optimizers/sgd.hpp"
#include "lbann/utils/cublas_wrapper.hpp"

namespace lbann {

namespace {

__global__ void momentum_kernel(DataType * __restrict__ values,
                                const DataType * __restrict__ gradient,
                                DataType * __restrict__ velocity,
                                int num_entries,
                                DataType learning_rate,
                                DataType momentum) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= num_entries) return;
  const DataType g = gradient[offset];
  DataType &v = velocity[offset];
  v = momentum * v + g;
  values[offset] -= learning_rate * v;
}

__global__ void nesterov_kernel(DataType * __restrict__ values,
                                const DataType * __restrict__ gradient,
                                DataType * __restrict__ velocity,
                                int num_entries,
                                DataType learning_rate,
                                DataType momentum) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= num_entries) return;
  const DataType g = gradient[offset];
  DataType &v = velocity[offset];
  v = momentum * v + g;
  values[offset] -= learning_rate * (momentum * v + g);
}

}

void sgd::step_compute_gpu(cudnn::matrix& values_d,
                           const cudnn::matrix& gradient_d) {

  // Get matrix dimensions
  const int num_entries = m_weights->get_size();
  if (num_entries == 0) return;

  // SGD without momentum
  if (m_momentum == DataType(0)) {
    for (int i = 0; i < m_cudnn->get_num_gpus(); ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUBLAS(cublas::axpy(m_cudnn->get_cublas_handle(i),
                                num_entries,
                                -m_learning_rate,
                                gradient_d.get_locked_data(i), 1,
                                values_d.get_data(i), 1));
    }
    return;
  }

  // Launch CUDA kernels for momentum SGD or NAG
  const int block_size = 256;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.x = (num_entries + block_size - 1) / block_size;
  for (int i = 0; i < m_cudnn->get_num_gpus(); ++i) {
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
    cudaStream_t stream = this->m_cudnn->get_stream(i);
    if (m_nesterov) {
      nesterov_kernel<<<grid_dims, block_dims, 0, stream>>>
        (values_d.get_data(i), gradient_d.get_locked_data(i),
         m_velocity_d[i], num_entries, m_learning_rate, m_momentum);
    } else {
      momentum_kernel<<<grid_dims, block_dims, 0, stream>>>
        (values_d.get_data(i), gradient_d.get_locked_data(i),
         m_velocity_d[i], num_entries, m_learning_rate, m_momentum);
    }
  }

}

}  // namespace lbann
