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

#include "lbann/optimizers/adagrad.hpp"

namespace lbann {

namespace {

// Square root functions
#if __CUDA_ARCH__ >= 530
__device__ inline float sqrt_(__half x) {
  return hsqrt(x);
}
#endif // __CUDA_ARCH__ >= 530
__device__ inline float sqrt_(float x) {
  return sqrtf(x);
}
__device__ inline double sqrt_(double x) {
  return sqrt(x);
}

__global__ void adagrad_kernel(int height,
                               int width,
                               DataType learning_rate,
                               DataType eps,
                               DataType * __restrict__ values,
                               int values_ldim,
                               const DataType * __restrict__ gradient,
                               int gradient_ldim,
                               DataType * __restrict__ cache,
                               int cache_ldim) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_threads = gridDim.x * blockDim.x;
  for (int pos = gid; pos < height * width; pos += num_threads) {
    const auto& i = pos % height;
    const auto& j = pos / height;
    auto& x = values[i + j * values_ldim];
    const auto& g = gradient[i + j * gradient_ldim];
    auto& c = cache[i + j * cache_ldim];
    c += g * g;
    x -= learning_rate * g / (sqrt_(c) + eps);
  }
}

} // namespace

void adagrad::step_compute_gpu(AbsDistMat& values, const AbsDistMat& gradient) {
  const int local_height = values.LocalHeight();
  const int local_width = values.LocalWidth();
  const int size = local_height * local_width;
  const int block_dim = 256;
  const int grid_dim = (size + block_dim - 1) / block_dim;
  if (grid_dim > 0) {
    adagrad_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
      local_height, local_width, m_learning_rate, m_eps,
      values.Buffer(), values.LDim(),
      gradient.LockedBuffer(), gradient.LDim(),
      m_cache->Buffer(), m_cache->LDim());
  }
}

} // namespace lbann
