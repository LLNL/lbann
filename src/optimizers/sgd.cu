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

template <typename TensorDataType>
__global__ void momentum_noncontiguous_kernel(size_t height,
                                              size_t width,
                                              TensorDataType learning_rate,
                                              TensorDataType momentum,
                                              TensorDataType * __restrict__ values,
                                              size_t values_ldim,
                                              const TensorDataType * __restrict__ gradient,
                                              size_t gradient_ldim,
                                              TensorDataType * __restrict__ velocity,
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

template <typename TensorDataType>
__global__ void momentum_contiguous_kernel(size_t size,
                                           TensorDataType learning_rate,
                                           TensorDataType momentum,
                                           TensorDataType * __restrict__ values,
                                           const TensorDataType * __restrict__ gradient,
                                           TensorDataType * __restrict__ velocity) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < size) {
    const auto& g = gradient[gid];
    auto& v = velocity[gid];
    auto& x = values[gid];
    v = momentum * v + g;
    x -= learning_rate * v;
  }
}

template <typename TensorDataType>
__global__ void nesterov_kernel(size_t height,
                                size_t width,
                                TensorDataType learning_rate,
                                TensorDataType momentum,
                                TensorDataType * __restrict__ values,
                                size_t values_ldim,
                                const TensorDataType * __restrict__ gradient,
                                size_t gradient_ldim,
                                TensorDataType * __restrict__ velocity,
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

template <typename TensorDataType>
void sgd<TensorDataType>::momentum_step_gpu(AbsDistMatrixType& values,
                                            const AbsDistMatrixType& gradient) {

  // Get matrix dimensions
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  const size_t local_size = local_height * local_width;
  if (local_size <= 0) { return; }

  // Launch GPU kernels for momentum SGD or NAG
  constexpr size_t block_size = 256;
  const size_t grid_size = (local_size + block_size - 1) / block_size;
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(values),
                                     gpu::get_sync_info(gradient));
  if (m_nesterov) {
    hydrogen::gpu::LaunchKernel(
      nesterov_kernel<TensorDataType>,
      grid_size, block_size, 0, multisync,
      local_height, local_width,
      this->get_learning_rate(), m_momentum,
      values.Buffer(), values.LDim(),
      gradient.LockedBuffer(), gradient.LDim(),
      m_velocity->Buffer(), m_velocity->LDim());
  } else {
    if (values.Contiguous() && gradient.Contiguous()
        && m_velocity->Contiguous()) {
      hydrogen::gpu::LaunchKernel(
        momentum_contiguous_kernel<TensorDataType>,
        grid_size, block_size, 0, multisync,
        local_size, this->get_learning_rate(), m_momentum,
        values.Buffer(), gradient.LockedBuffer(), m_velocity->Buffer());
    } else {
      hydrogen::gpu::LaunchKernel(
        momentum_noncontiguous_kernel<TensorDataType>,
        grid_size, block_size, 0, multisync,
        local_height, local_width,
        this->get_learning_rate(), m_momentum,
        values.Buffer(), values.LDim(),
        gradient.LockedBuffer(), gradient.LDim(),
        m_velocity->Buffer(), m_velocity->LDim());
    }
  }

}

#ifdef LBANN_HAS_HALF
template <>
void sgd<cpu_fp16>::momentum_step_gpu(AbsDistMatrixType&,
                                      const AbsDistMatrixType&) {
  LBANN_ERROR("Can't call this function with cpu_fp16!");
}
#endif // LBANN_HAS_HALF

#define PROTO(T)                            \
  template void sgd<T>::momentum_step_gpu(  \
    El::AbstractDistMatrix<T>&,             \
    const El::AbstractDistMatrix<T>&)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
} // namespace lbann
