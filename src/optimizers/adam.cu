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

#include "lbann/optimizers/adam.hpp"

namespace lbann {

namespace {

__global__ void adam_noncontiguous_kernel(size_t height,
                                          size_t width,
                                          DataType correction,
                                          DataType eps,
                                          DataType beta1,
                                          DataType beta2,
                                          DataType * __restrict__ values,
                                          size_t values_ldim,
                                          const DataType * __restrict__ gradient,
                                          size_t gradient_ldim,
                                          DataType * __restrict__ moment1,
                                          size_t moment1_ldim,
                                          DataType * __restrict__ moment2,
                                          size_t moment2_ldim) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < height * width) {
    const auto& row = gid % height;
    const auto& col = gid / height;
    const auto& g = gradient[row + col * gradient_ldim] + eps;
    auto& m1 = moment1[row + col * moment1_ldim];
    auto& m2 = moment2[row + col * moment2_ldim];
    auto& x = values[row + col * values_ldim];
    m1 = beta1 * m1 + (DataType(1) - beta1) * g;
    m2 = beta2 * m2 + (DataType(1) - beta2) * g * g;
    x -= correction * m1 / (cuda::sqrt(m2) + eps);
  }
}

__global__ void adam_contiguous_kernel(size_t size,
                                       DataType correction,
                                       DataType eps,
                                       DataType beta1,
                                       DataType beta2,
                                       DataType * __restrict__ values,
                                       const DataType * __restrict__ gradient,
                                       DataType * __restrict__ moment1,
                                       DataType * __restrict__ moment2) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < size) {
    const auto& g = gradient[gid] + eps;
    auto& m1 = moment1[gid];
    auto& m2 = moment2[gid];
    auto& x = values[gid];
    m1 = beta1 * m1 + (DataType(1) - beta1) * g;
    m2 = beta2 * m2 + (DataType(1) - beta2) * g * g;
    x -= correction * m1 / (cuda::sqrt(m2) + eps);
  }
}

} // namespace

void adam::step_compute_gpu(AbsDistMat& values, const AbsDistMat& gradient) {
  constexpr DataType one = 1;

  // Precompute the bias correction and learning rate.
  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  const DataType correction = this->get_learning_rate() *
                              (std::sqrt(one - m_current_beta2)
                               / (one - m_current_beta1));

  // Get matrix dimensions
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  const size_t local_size = local_height * local_width;
  if (local_size <= 0) { return; }

  // Launch CUDA kernel
  constexpr size_t block_size = 256;
  const size_t grid_size = (local_size + block_size - 1) / block_size;
  auto&& stream = El::GPUManager::Stream();
  if (values.Contiguous() && gradient.Contiguous()
      && m_moment1->Contiguous() && m_moment2->Contiguous()) {
    adam_contiguous_kernel<<<grid_size, block_size, 0, stream>>>(
      local_size, correction, m_eps, m_beta1, m_beta2,
      values.Buffer(), gradient.LockedBuffer(),
      m_moment1->Buffer(), m_moment2->Buffer());
  } else {
    adam_noncontiguous_kernel<<<grid_size, block_size, 0, stream>>>(
      local_height, local_width, correction, m_eps, m_beta1, m_beta2,
      values.Buffer(), values.LDim(),
      gradient.LockedBuffer(), gradient.LDim(),
      m_moment1->Buffer(), m_moment1->LDim(),
      m_moment2->Buffer(), m_moment2->LDim());
  }

}

} // namespace lbann
