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

namespace lbann {

namespace {

__global__ void momentum_kernel(int height,
                                int width,
                                DataType learning_rate,
                                DataType momentum,
                                DataType * __restrict__ values,
                                int values_ldim,
                                const DataType * __restrict__ gradient,
                                int gradient_ldim,
                                DataType * __restrict__ velocity,
                                int velocity_ldim) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_threads = gridDim.x * blockDim.x;
  for (int pos = tid; pos < height * width; pos += num_threads) {
    const auto& i = pos % height;
    const auto& j = pos / height;
    const auto& g = gradient[i + j * gradient_ldim];
    auto& v = velocity[i + j * velocity_ldim];
    auto& x = values[i + j * values_ldim];
    v = momentum * v + g;
    x -= learning_rate * v;
  }
}

__global__ void nesterov_kernel(int height,
                                int width,
                                DataType learning_rate,
                                DataType momentum,
                                DataType * __restrict__ values,
                                int values_ldim,
                                const DataType * __restrict__ gradient,
                                int gradient_ldim,
                                DataType * __restrict__ velocity,
                                int velocity_ldim) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_threads = gridDim.x * blockDim.x;
  for (int pos = tid; pos < height * width; pos += num_threads) {
    const auto& i = pos % height;
    const auto& j = pos / height;
    const auto& g = gradient[i + j * gradient_ldim];
    auto& v = velocity[i + j * velocity_ldim];
    auto& x = values[i + j * values_ldim];
    v = momentum * v + g;
    x -= learning_rate * (momentum * v + g);
  }
}

}

void sgd::step_compute_gpu(AbsDistMat& values, const AbsDistMat& gradient) {

  // Get matrix dimensions
  const int local_height = values.LocalHeight();
  const int local_width = values.LocalWidth();
  const int size = local_height * local_width;
  if (size <= 0) { return; }

  // SGD without momentum
  if (m_momentum == DataType(0)) {
    El::Axpy(-m_learning_rate, gradient, values);
    return;
  }

  // Launch CUDA kernels for momentum SGD or NAG
  const int block_size = 256;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.x = (size + block_size - 1) / block_size;
  cudaStream_t stream = El::GPUManager::Stream();
  if (m_nesterov) {
    nesterov_kernel<<<grid_dims, block_dims, 0, stream>>>
      (local_height, local_width, m_learning_rate, m_momentum,
       values.Buffer(), values.LDim(),
       gradient.LockedBuffer(), gradient.LDim(),
       m_velocity->Buffer(), m_velocity->LDim());
  } else {
    momentum_kernel<<<grid_dims, block_dims, 0, stream>>>
      (local_height, local_width, m_learning_rate, m_momentum,
       values.Buffer(), values.LDim(),
       gradient.LockedBuffer(), gradient.LDim(),
       m_velocity->Buffer(), m_velocity->LDim());
  }

}

}  // namespace lbann
