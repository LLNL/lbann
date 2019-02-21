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

#include "lbann/optimizers/rmsprop.hpp"
#include "lbann/utils/cuda.hpp"

namespace lbann {

namespace {

__global__ void rmsprop_kernel(El::Int height,
                               El::Int width,
                               DataType learning_rate,
                               DataType decay_rate,
                               DataType eps,
                               DataType * __restrict__ values,
                               El::Int values_ldim,
                               const DataType * __restrict__ gradient,
                               El::Int gradient_ldim,
                               DataType * __restrict__ cache,
                               El::Int cache_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int nthreads = gridDim.x * blockDim.x;
  for (El::Int pos = gid; pos < height * width; pos += nthreads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& g = gradient[row + col * gradient_ldim];
    auto& c = cache[row + col * cache_ldim];
    auto& x = values[row + col * values_ldim];
    c = decay_rate * c + (DataType(1) - decay_rate) * g * g;
    x -= learning_rate * g / (cuda::sqrt(c) + eps);
  }
}

} // namespace

void rmsprop::step_compute_gpu(AbsDistMat& values, const AbsDistMat& gradient) {
  const El::Int local_height = values.LocalHeight();
  const El::Int local_width = values.LocalWidth();
  const El::Int size = local_height * local_width;
  constexpr El::Int block_dim = 256;
  const El::Int grid_dim = (size + block_dim - 1) / block_dim;
  if (grid_dim > 0) {
    rmsprop_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
      local_height, local_width, m_learning_rate, m_decay_rate, m_eps,
      values.Buffer(), values.LDim(),
      gradient.LockedBuffer(), gradient.LDim(),
      m_cache->Buffer(), m_cache->LDim());
  }
}

} // namespace lbann
