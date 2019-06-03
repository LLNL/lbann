////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

__global__ void momentum_noncontiguous_kernel(size_t height,
                                              size_t width,
                                              DataType learning_rate,
                                              DataType momentum,
                                              DataType * __restrict__ values,
                                              size_t values_ldim,
                                              const DataType * __restrict__ gradient,
                                              size_t gradient_ldim,
                                              DataType * __restrict__ velocity,
                                              size_t velocity_ldim) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < height * width) {
    const auto& row = gid % height;
    const auto& col = gid / height;
    const auto& g = gradient[row + col * gradient_ldim];
    auto& v = velocity[row + col * velocity_ldim];
    auto& x = values[row + col * values_ldim];
    v = momentum * v + g;
    x -= learning_rate * v;
  }
}

__global__ void momentum_contiguous_kernel(size_t size,
                                           DataType learning_rate,
                                           DataType momentum,
                                           DataType * __restrict__ values,
                                           const DataType * __restrict__ gradient,
                                           DataType * __restrict__ velocity) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < size) {
    const auto& g = gradient[gid];
    auto& v = velocity[gid];
    auto& x = values[gid];
    v = momentum * v + g;
    x -= learning_rate * v;
  }
}

__global__ void nesterov_kernel(size_t height,
                                size_t width,
                                DataType learning_rate,
                                DataType momentum,
                                DataType * __restrict__ values,
                                size_t values_ldim,
                                const DataType * __restrict__ gradient,
                                size_t gradient_ldim,
                                DataType * __restrict__ velocity,
                                size_t velocity_ldim) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = gridDim.x * blockDim.x;
  for (size_t pos = gid; pos < height * width; pos += nthreads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& g = gradient[row + col * gradient_ldim];
    auto& v = velocity[row + col * velocity_ldim];
    auto& x = values[row + col * values_ldim];
    v = momentum * v + g;
    x -= learning_rate * (momentum * v + g);
  }
}

} // namespace

void sgd::momentum_step_gpu(AbsDistMat& values, const AbsDistMat& gradient) {

  // Get matrix dimensions
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  const size_t local_size = local_height * local_width;
  if (local_size <= 0) { return; }

  // Launch CUDA kernels for momentum SGD or NAG
  constexpr size_t block_size = 256;
  const size_t grid_size = (local_size + block_size - 1) / block_size;
  auto&& stream = El::GPUManager::Stream();
  if (m_nesterov) {
    nesterov_kernel<<<grid_size, block_size, 0, stream>>>(
      local_height, local_width,
      this->get_learning_rate(), m_momentum,
      values.Buffer(), values.LDim(),
      gradient.LockedBuffer(), gradient.LDim(),
      m_velocity->Buffer(), m_velocity->LDim());
  } else {
    if (values.Contiguous() && gradient.Contiguous()
        && m_velocity->Contiguous()) {
      momentum_contiguous_kernel<<<grid_size, block_size, 0, stream>>>(
        local_size, this->get_learning_rate(), m_momentum,
        values.Buffer(), gradient.LockedBuffer(), m_velocity->Buffer());
    } else {
      momentum_noncontiguous_kernel<<<grid_size, block_size, 0, stream>>>(
        local_height, local_width,
        this->get_learning_rate(), m_momentum,
        values.Buffer(), values.LDim(),
        gradient.LockedBuffer(), gradient.LDim(),
        m_velocity->Buffer(), m_velocity->LDim());
    }
  }

}

} // namespace lbann
