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

#define LBANN_ROWWISE_WEIGHTS_NORMS_LAYER_INSTANTIATE
#include "lbann/layers/misc/rowwise_weights_norms_impl.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

/**
 *  Block dimensions: bdimx x bdimy x 1
 *
 *  Grid dimensions: (height/bdimx) x (width/bdimy) x 1
 */
template <typename T>
__global__ void row_sqsums_kernel(size_t height,
                                  size_t width,
                                  const T* __restrict__ mat,
                                  size_t mat_ldim,
                                  T* __restrict__ row_sqsums)
{

  // Indices
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;

  // Accumulate sum of squares for each matrix row
  for (size_t row = gidx; row < height; row += nthreadsx) {
    T sqsum{0};
    for (size_t col = gidy; col < width; col += nthreadsy) {
      const auto& x = mat[row + col * mat_ldim];
      sqsum += x * x;
    }
    gpu_lib::atomic_add(&row_sqsums[row], sqsum);
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::row_sqsums(
  const El::Matrix<TensorDataType, Device>& mat,
  El::Matrix<TensorDataType, Device>& row_sqsums)
{
  LBANN_CALIPER_MARK_FUNCTION;

  // Launch kernel
  El::Zero(row_sqsums);
  if (!mat.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(row_sqsums),
                                       gpu::get_sync_info(mat));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    block_dims.y = 1;
    grid_dims.x = (mat.Height() + block_dims.x - 1) / block_dims.x;
    grid_dims.y = (mat.Width() / 64 + block_dims.y - 1) / block_dims.y;
    grid_dims.y = El::Min(El::Max(grid_dims.y, 1), 65536);
    hydrogen::gpu::LaunchKernel(row_sqsums_kernel<TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                static_cast<size_t>(mat.Height()),
                                static_cast<size_t>(mat.Width()),
                                mat.LockedBuffer(),
                                static_cast<size_t>(mat.LDim()),
                                row_sqsums.Buffer());
  }
}

namespace {

/**
 *  Block dimensions: bdim x 1 x 1
 *
 *  Grid dimensions: (size/bdim) x 1 x 1
 */
template <typename T>
__global__ void sqrt_kernel(size_t size, T* __restrict__ buf)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  for (size_t i = gid; i < size; i += nthreads) {
    auto& x = buf[i];
    x = gpu_lib::sqrt(x);
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::sqrt(
  El::Matrix<TensorDataType, Device>& mat)
{
  LBANN_CALIPER_MARK_FUNCTION;

  // Check that matrix is valid
  if (!mat.Contiguous()) {
    LBANN_ERROR("matrix is not contiguous");
  }

  // Launch kernel
  if (!mat.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (mat.Height() * mat.Width() + block_size - 1) / block_size;
    hydrogen::gpu::LaunchKernel(sqrt_kernel<TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                gpu::get_sync_info(mat),
                                static_cast<size_t>(mat.Height() * mat.Width()),
                                mat.Buffer());
  }
}

namespace {

/**
 *  Block dimensions: bdim x 1 x 1
 *
 *  Grid dimensions: (size/bdim) x 1 x 1
 */
template <typename T>
__global__ void
divide_kernel(size_t size, T* __restrict__ numer, const T* __restrict__ denom)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  for (size_t i = gid; i < size; i += nthreads) {
    auto& x = numer[i];
    const auto& y = denom[i];
    const auto& z = x / y;
    x = gpu_lib::isfinite(z) ? z : T{0};
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::divide(
  El::Matrix<TensorDataType, Device>& numer,
  const El::Matrix<TensorDataType, Device>& denom)
{
  LBANN_CALIPER_MARK_FUNCTION;

  // Check that matrices are valid
  if (numer.Height() != denom.Height() || numer.Width() != denom.Width()) {
    LBANN_ERROR("numerator and denominator do not have same dims");
  }
  if (!numer.Contiguous() || !denom.Contiguous()) {
    LBANN_ERROR("matrices are not contiguous");
  }

  // Launch kernel
  if (!numer.IsEmpty()) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(numer), gpu::get_sync_info(denom));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x =
      (numer.Height() * numer.Width() + block_size - 1) / block_size;
    hydrogen::gpu::LaunchKernel(divide_kernel<TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                numer.Height() * numer.Width(),
                                numer.Buffer(),
                                denom.LockedBuffer());
  }
}

namespace {

template <typename T>
__global__ void row_axpy_kernel(size_t height,
                                size_t width,
                                T alpha,
                                const T* __restrict__ a_vec,
                                const T* __restrict__ x_mat,
                                size_t x_ldim,
                                T beta,
                                T* __restrict__ y_mat,
                                size_t y_ldim)
{

  // Indices
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;

  // Accumulate sum of squares for each matrix row
  for (size_t row = gidx; row < height; row += nthreadsx) {
    const auto& alpha_a = alpha * a_vec[row];
    for (size_t col = gidy; col < width; col += nthreadsy) {
      const auto& x = x_mat[row + col * x_ldim];
      auto& y = y_mat[row + col * y_ldim];
      y = alpha_a * x + beta * y;
    }
  }
}

} // namespace

/**
 *  Block dimensions: bdimx x bdimy x 1
 *
 *  Grid dimensions: (height/bdimx) x (width/bdimy) x 1
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::row_axpy(
  TensorDataType alpha,
  const El::Matrix<TensorDataType, Device>& a_vec,
  const El::Matrix<TensorDataType, Device>& x_mat,
  TensorDataType beta,
  El::Matrix<TensorDataType, Device>& y_mat)
{
  LBANN_CALIPER_MARK_FUNCTION;

  // Check that matrices are valid
  if (x_mat.Height() != y_mat.Height() || x_mat.Width() != y_mat.Width() ||
      a_vec.Height() != y_mat.Height() || a_vec.Width() != 1) {
    LBANN_ERROR("matrix dims do not match");
  }

  // Launch kernel
  if (!y_mat.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(y_mat),
                                       gpu::get_sync_info(a_vec),
                                       gpu::get_sync_info(x_mat));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    block_dims.x = 1;
    grid_dims.x = (y_mat.Height() + block_dims.x - 1) / block_dims.x;
    grid_dims.y = (y_mat.Width() + block_dims.y - 1) / block_dims.y;
    grid_dims.y = El::Min(grid_dims.y, 65536);
    hydrogen::gpu::LaunchKernel(row_axpy_kernel<TensorDataType>,
                                grid_dims,
                                block_dims,
                                0,
                                multisync,
                                static_cast<size_t>(y_mat.Height()),
                                static_cast<size_t>(y_mat.Width()),
                                alpha,
                                a_vec.LockedBuffer(),
                                x_mat.LockedBuffer(),
                                static_cast<size_t>(x_mat.LDim()),
                                beta,
                                y_mat.Buffer(),
                                static_cast<size_t>(y_mat.LDim()));
  }
}

#define PROTO(T)                                                               \
  template class rowwise_weights_norms_layer<T,                                \
                                             data_layout::DATA_PARALLEL,       \
                                             El::Device::GPU>;                 \
  template class rowwise_weights_norms_layer<T,                                \
                                             data_layout::MODEL_PARALLEL,      \
                                             El::Device::GPU>
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
