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

#include "lbann/optimizers/rmsprop.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
__global__ void rmsprop_kernel(size_t height,
                               size_t width,
                               TensorDataType learning_rate,
                               TensorDataType decay_rate,
                               TensorDataType eps,
                               TensorDataType * __restrict__ values,
                               size_t values_ldim,
                               const TensorDataType * __restrict__ gradient,
                               size_t gradient_ldim,
                               TensorDataType * __restrict__ cache,
                               size_t cache_ldim) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = gridDim.x * blockDim.x;
  for (size_t pos = gid; pos < height * width; pos += nthreads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& g = gradient[row + col * gradient_ldim];
    auto& c = cache[row + col * cache_ldim];
    auto& x = values[row + col * values_ldim];
    c = decay_rate * c + (TensorDataType(1) - decay_rate) * g * g;
    x -= learning_rate * g / (cuda::sqrt(c) + eps);
  }
}

} // namespace

template <typename TensorDataType>
void rmsprop<TensorDataType>::step_compute_gpu(AbsDistMatrixType& values,
                                               const AbsDistMatrixType& gradient) {
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  const size_t local_size = local_height * local_width;
  if (local_size > 0) {
    constexpr size_t block_size = 256;
    const size_t grid_size = (local_size + block_size - 1) / block_size;
    auto multisync = hydrogen::MakeMultiSync(gpu::get_sync_info(values),
                                             gpu::get_sync_info(gradient));
    hydrogen::gpu::LaunchKernel(
      rmsprop_kernel<TensorDataType>, grid_size, block_size, 0,
      static_cast<El::SyncInfo<El::Device::GPU> const&>(multisync),
      local_height, local_width,
      this->get_learning_rate(), m_decay_rate, m_eps,
      values.Buffer(), values.LDim(),
      gradient.LockedBuffer(), gradient.LDim(),
      m_cache->Buffer(), m_cache->LDim());
  }
}

#ifdef LBANN_HAS_HALF
template <>
void rmsprop<cpu_fp16>::step_compute_gpu(AbsDistMatrixType&,
                                         const AbsDistMatrixType&) {
  LBANN_ERROR("Can't call this function with cpu_fp16!");
}
#endif // LBANN_HAS_HALF

#define PROTO(T)                               \
  template void rmsprop<T>::step_compute_gpu(  \
    El::AbstractDistMatrix<T>&,                \
    const El::AbstractDistMatrix<T>&)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
