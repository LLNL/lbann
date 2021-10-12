////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
__global__ void
adam_noncontiguous_kernel(size_t height,
                          size_t width,
                          TensorDataType correction,
                          TensorDataType eps,
                          TensorDataType beta1,
                          TensorDataType beta2,
                          TensorDataType* __restrict__ values,
                          size_t values_ldim,
                          const TensorDataType* __restrict__ gradient,
                          size_t gradient_ldim,
                          TensorDataType* __restrict__ moment1,
                          size_t moment1_ldim,
                          TensorDataType* __restrict__ moment2,
                          size_t moment2_ldim)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < height * width) {
    const auto& row = gid % height;
    const auto& col = gid / height;
    const auto& g = gradient[row + col * gradient_ldim] + eps;
    if (gpu_lib::isinf(g) || gpu_lib::isnan(g)) {
      return;
    }
    auto& m1 = moment1[row + col * moment1_ldim];
    auto& m2 = moment2[row + col * moment2_ldim];
    auto& x = values[row + col * values_ldim];
    m1 = beta1 * m1 + (TensorDataType(1) - beta1) * g;
    m2 = beta2 * m2 + (TensorDataType(1) - beta2) * g * g;
    x -= correction * m1 / (gpu_lib::sqrt(m2) + eps);
  }
}

template <typename TensorDataType>
__global__ void
adam_contiguous_kernel(size_t size,
                       TensorDataType correction,
                       TensorDataType eps,
                       TensorDataType beta1,
                       TensorDataType beta2,
                       TensorDataType* __restrict__ values,
                       const TensorDataType* __restrict__ gradient,
                       TensorDataType* __restrict__ moment1,
                       TensorDataType* __restrict__ moment2)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < size) {
    const auto& g = gradient[gid] + eps;
    if (gpu_lib::isinf(g) || gpu_lib::isnan(g)) {
      return;
    }
    auto& m1 = moment1[gid];
    auto& m2 = moment2[gid];
    auto& x = values[gid];
    m1 = beta1 * m1 + (TensorDataType(1) - beta1) * g;
    m2 = beta2 * m2 + (TensorDataType(1) - beta2) * g * g;
    x -= correction * m1 / (gpu_lib::sqrt(m2) + eps);
  }
}

} // namespace

template <typename TensorDataType>
void adam<TensorDataType>::step_compute_gpu(AbsDistMatrixType& values,
                                            const AbsDistMatrixType& gradient,
                                            const TensorDataType& correction)
{
  LBANN_CALIPER_MARK_SCOPE("adam::step_compute");
  // Get matrix dimensions
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  const size_t local_size = local_height * local_width;
  if (local_size <= 0) {
    return;
  }

  // Launch GPU kernel
  constexpr size_t block_size = 256;
  const size_t grid_size = (local_size + block_size - 1) / block_size;
  auto multisync =
    El::MakeMultiSync(gpu::get_sync_info(values), gpu::get_sync_info(gradient));
  if (values.Contiguous() && gradient.Contiguous() && m_moment1->Contiguous() &&
      m_moment2->Contiguous()) {
    hydrogen::gpu::LaunchKernel(adam_contiguous_kernel<TensorDataType>,
                                grid_size,
                                block_size,
                                0,
                                multisync,
                                local_size,
                                correction,
                                m_eps,
                                m_beta1,
                                m_beta2,
                                values.Buffer(),
                                gradient.LockedBuffer(),
                                m_moment1->Buffer(),
                                m_moment2->Buffer());
  }
  else {
    hydrogen::gpu::LaunchKernel(adam_noncontiguous_kernel<TensorDataType>,
                                grid_size,
                                block_size,
                                0,
                                multisync,
                                local_height,
                                local_width,
                                correction,
                                m_eps,
                                m_beta1,
                                m_beta2,
                                values.Buffer(),
                                values.LDim(),
                                gradient.LockedBuffer(),
                                gradient.LDim(),
                                m_moment1->Buffer(),
                                m_moment1->LDim(),
                                m_moment2->Buffer(),
                                m_moment2->LDim());
  }
}

#ifdef LBANN_HAS_HALF
template <>
void adam<cpu_fp16>::step_compute_gpu(AbsDistMatrixType&,
                                      const AbsDistMatrixType&,
                                      const cpu_fp16&)
{
  LBANN_ERROR("Can't call this function with cpu_fp16!");
}
#endif // LBANN_HAS_HALF

#define PROTO(T)                                                               \
  template void adam<T>::step_compute_gpu(El::AbstractDistMatrix<T>&,          \
                                          const El::AbstractDistMatrix<T>&,    \
                                          const T&)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
