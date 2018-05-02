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

#include "lbann/optimizers/adam.hpp"

namespace lbann {

namespace {

__global__ void adam_kernel(int height,
                            int width,
                            DataType correction,
                            DataType eps,
                            DataType beta1,
                            DataType beta2,
                            DataType * __restrict__ values,
                            int values_ldim,
                            const DataType * __restrict__ gradient,
                            int gradient_ldim,
                            DataType * __restrict__ moment1,
                            int moment1_ldim,
                            DataType * __restrict__ moment2,
                            int moment2_ldim) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_threads = gridDim.x * blockDim.x;
  for (int pos = tid; pos < height * width; pos += num_threads) {
    const auto& i = pos % height;
    const auto& j = pos / height;
    const auto& g = gradient[i + j * gradient_ldim] + eps;
    auto& m1 = moment1[i + j * moment1_ldim];
    auto& m2 = moment2[i + j * moment2_ldim];
    auto& x = values[i + j * values_ldim];
    m1 = beta1 * m1 + (DataType(1) - beta1) * g;
    m2 = beta2 * m2 + (DataType(1) - beta2) * g * g;
    x -= correction * m1 / (sqrt(m2) + eps);
  }
}

}

void adam::step_compute_gpu(AbsDistMat& values, const AbsDistMat& gradient) {

  // Precompute the bias correction and learning rate.
  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  const DataType correction = m_learning_rate *
                              (std::sqrt(DataType(1) - m_current_beta2)
                               / (DataType(1) - m_current_beta1));

  // Get matrix dimensions
  const int local_height = values.LocalHeight();
  const int local_width = values.LocalWidth();
  const int size = local_height * local_width;
  if (size <= 0) { return; }

  // Launch CUDA kernels
  const int block_size = 256;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.x = (size + block_size - 1) / block_size;
  CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
  cudaStream_t stream = this->m_cudnn->get_stream();
  adam_kernel<<<grid_dims, block_dims, 0, stream>>>
    (local_height, local_width, correction, m_eps, m_beta1, m_beta2,
     values.Buffer(), values.LDim(),
     gradient.LockedBuffer(), gradient.LDim(),
     m_moment1->Buffer(), m_moment1->LDim(),
     m_moment2->Buffer(), m_moment2->LDim());

}

}  // namespace lbann
