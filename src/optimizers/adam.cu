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
// adam .hpp .cpp .cu - SGD with Adam
// Reference:
// Kingma, D. and Ba, J. 2014. Adam: A Method for Stochastic Optimization.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/adam.hpp"

namespace lbann {

namespace {

__global__ void adam_kernel(DataType * __restrict__ values,
                            const DataType * __restrict__ gradient,
                            DataType * __restrict__ moment1,
                            DataType * __restrict__ moment2,
                            int num_entries,
                            DataType correction,
                            DataType eps,
                            DataType beta1,
                            DataType beta2) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= num_entries) return;
  const DataType g = gradient[offset] + eps;
  DataType &m1 = moment1[offset];
  DataType &m2 = moment2[offset];
  m1 = beta1 * m1 + (DataType(1) - beta1) * g;
  m2 = beta2 * m2 + (DataType(1) - beta2) * g * g;
  values[offset] -= correction * m1 / (sqrt(m2) + eps);
}

}

void adam::step_compute_gpu(std::vector<DataType*> values_d,
                            std::vector<DataType*> gradient_d) {

  // Precompute the bias correction and learning rate.
  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  const DataType correction = m_learning_rate *
                              (std::sqrt(DataType(1) - m_current_beta2)
                               / (DataType(1) - m_current_beta1));

  // Get matrix dimensions
  const int num_entries = m_weights->get_size();
  if (num_entries == 0) return;

  // Launch CUDA kernels
  const int block_size = 256;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.x = (num_entries + block_size - 1) / block_size;
  for (int i = 0; i < m_cudnn->get_num_gpus(); ++i) {
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
    cudaStream_t stream = this->m_cudnn->get_stream(i);
    adam_kernel<<<grid_dims, block_dims, 0, stream>>>
      (values_d[i], gradient_d[i], m_moment1_d[i], m_moment2_d[i],
       num_entries, correction, m_eps, m_beta1, m_beta2);
  }

}

}  // namespace lbann
