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

#include "lbann/utils/amp.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {
namespace amp {

namespace {

// std::isfinite does not work for half precision, so we handle that here.
// There does not seem to be a CUDA/ROCm version of isfinite for __half, so
// we handle it by converting to float.

template <typename T>
__host__ __device__ __forceinline__ bool amp_is_finite(T val)
{
  return std::isfinite(val);
}

#ifdef LBANN_HAS_GPU_FP16
template <>
__host__ __device__ __forceinline__ bool amp_is_finite<fp16>(fp16 val)
{
  return std::isfinite(static_cast<float>(val));
}
#endif

template <typename TensorDataType>
__global__ void is_finite_and_unscale_contiguous_kernel(
  size_t size,
  TensorDataType* __restrict__ grads,
  const TensorDataType inv_scale,
  float* is_finite) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < size) {
    TensorDataType& val = grads[gid];
    if (!amp_is_finite<TensorDataType>(val)) {
      *is_finite = 0.0f;
    }
    val *= inv_scale;
  }
}

template <typename TensorDataType>
__global__ void is_finite_and_unscale_noncontiguous_kernel(
  size_t height,
  size_t width,
  size_t ldim,
  TensorDataType* __restrict__ grads,
  const TensorDataType inv_scale,
  float* is_finite) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < height * width) {
    const auto row = gid % height;
    const auto col = gid / height;
    TensorDataType& val = grads[row + col*ldim];
    if (!amp_is_finite<TensorDataType>(val)) {
      *is_finite = 0.0f;
    }
    val *= inv_scale;
  }
}

}  // anonymous namespace

template <typename TensorDataType>
void is_finite_and_unscale_gpu(
  El::AbstractDistMatrix<TensorDataType>& grads,
  EvalType scale,
  float* is_finite) {
  LBANN_CALIPER_MARK_SCOPE("amp::is_finite_and_unscale");

  const size_t height = grads.LocalHeight();
  const size_t width = grads.LocalWidth();
  const size_t size = height * width;
  constexpr size_t block_size = 256;
  const size_t grid_size = (size + block_size - 1) / block_size;
  const TensorDataType inv_scale = El::To<TensorDataType>(EvalType(1) / scale);

  if (grads.Contiguous()) {
    hydrogen::gpu::LaunchKernel(
      is_finite_and_unscale_contiguous_kernel<TensorDataType>,
      grid_size,
      block_size,
      0,
      gpu::get_sync_info(grads),
      size,
      grads.Buffer(),
      inv_scale,
      is_finite);
  } else {
    hydrogen::gpu::LaunchKernel(
      is_finite_and_unscale_noncontiguous_kernel<TensorDataType>,
      grid_size,
      block_size,
      0,
      gpu::get_sync_info(grads),
      height,
      width,
      grads.LDim(),
      grads.Buffer(),
      inv_scale,
      is_finite);
  }
}

#ifdef LBANN_HAS_HALF
template <>
void is_finite_and_unscale_gpu<cpu_fp16>(El::AbstractDistMatrix<cpu_fp16>&, EvalType, float*) {
  LBANN_ERROR("Do not call the GPU kernels with cpu_fp16!");
}
#endif

#define PROTO(T)                                        \
  template void is_finite_and_unscale_gpu<T>(           \
    El::AbstractDistMatrix<T>& grads,                   \
    EvalType scale,                                     \
    float* is_finite);

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace amp
}  // namespace lbann
