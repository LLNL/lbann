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
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/execution_algorithms/kfac/kfac_util.hpp"

namespace lbann {
namespace kfac {

namespace {

template <typename TensorDataType>
__global__ void kfac_add_to_diagonal_kernel(TensorDataType* __restrict__ A,
                                            const size_t height,
                                            const TensorDataType value,
                                            const TensorDataType value_bn_err,
                                            const bool is_bn)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < height) {
    A[gid + gid * height] +=
      (is_bn && gid >= height / 2 ? value_bn_err : value);
  }
}

template <typename TensorDataType>
__global__ void kfac_make_diagonal_kernel(TensorDataType* __restrict__ A,
                                          TensorDataType* __restrict__ B,
                                          const size_t height,
                                          const TensorDataType value,
                                          const TensorDataType value_bn_err,
                                          const bool is_bn)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < height) {
    // A[gid+gid*height] = B[gid] + (is_bn && gid >= height/2 ? value_bn_err :
    // value);
    A[gid + gid * height] = B[gid] + value;

    // A[gid+gid*height] = gpu_lib::pow(A[gid+gid*height],-1);
    A[gid + gid * height] = TensorDataType(1) / A[gid + gid * height];
  }
}

template <typename TensorDataType>
__global__ void kfac_fill_upper_tri_kernel(TensorDataType* __restrict__ A,
                                           const size_t height)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t row = gid % height, col = gid / height;
  if (row < height && col < height && row < col) {
    A[row + col * height] += A[col + row * height];
  }
}

template <typename TensorDataType>
__global__ void
kfac_update_kronecker_average_kernel(TensorDataType* __restrict__ Aave,
                                     const TensorDataType* __restrict__ A,
                                     const size_t count,
                                     const double decay)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < count) {
    Aave[gid] = Aave[gid] * decay + A[gid] * (1.0 - decay);
  }
}

// This is never used because KFAC is only instantiated at `DataType`
// (i.e., "float" for most folks). I assume this should stay put for a
// future universe in which we perhaps consider other compute types?
#if 0  // def LBANN_HAS_HALF
template <>
__global__ void kfac_update_kronecker_average_kernel<__half>(
    __half * __restrict__ Aave,
    const __half * __restrict__ A,
    const size_t count,
    const double decay) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < count) {
    Aave[gid] = (float) Aave[gid]*decay + (float) A[gid]*(1.0-decay);
  }
}
#endif // LBANN_HAS_HALF

template <typename TensorDataType>
__global__ void kfac_identity_kernel(TensorDataType* __restrict__ A,
                                     const size_t height)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < height * height) {
    const size_t row = gid % height;
    const size_t col = gid / height;
    A[gid] = (row == col ? 1.0 : 0.0);
  }
}

template <typename TensorDataType>
__global__ void kfac_pack_lower_tri_kernel(TensorDataType* __restrict__ L,
                                           const TensorDataType* __restrict__ A,
                                           const size_t height)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < height * height) {
    const size_t row = gid % height;
    const size_t col = gid / height;
    if (row >= col) {
      L[row + (2 * height - (col - 1)) * col / 2 - col] = A[gid];
    }
  }
}

template <typename TensorDataType>
__global__ void
kfac_unpack_lower_tri_kernel(TensorDataType* __restrict__ A,
                             const TensorDataType* __restrict__ L,
                             const size_t height)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < height * height) {
    const size_t row = gid % height;
    const size_t col = gid / height;
    if (row >= col) {
      A[gid] = A[col + row * height] =
        L[row + (2 * height - (col - 1)) * col / 2 - col];
    }
  }
}

} // namespace

template <>
void add_to_diagonal(El::Matrix<DataType, El::Device::GPU>& A,
                     const DataType damping,
                     const DataType damping_bn_err,
                     const bool is_bn,
                     const El::SyncInfo<El::Device::GPU>& sync_info)
{
  const size_t height = A.Height();
  constexpr size_t block_size = 256;
  const size_t grid_size = (height + block_size - 1) / block_size;
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(kfac_add_to_diagonal_kernel<DataType>,
                                grid_size,
                                block_size,
                                0,
                                sync_info,
                                A.Buffer(),
                                height,
                                damping,
                                damping_bn_err,
                                is_bn);
  }
}

template <>
void make_diagonal(El::Matrix<DataType, El::Device::GPU>& A,
                   El::Matrix<DataType, El::Device::GPU>& B,
                   const DataType damping,
                   const DataType damping_bn_err,
                   const bool is_bn,
                   const El::SyncInfo<El::Device::GPU>& sync_info)
{
  const size_t height = A.Height();
  constexpr size_t block_size = 256;
  const size_t grid_size = (height + block_size - 1) / block_size;
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(kfac_make_diagonal_kernel<DataType>,
                                grid_size,
                                block_size,
                                0,
                                sync_info,
                                A.Buffer(),
                                B.Buffer(),
                                height,
                                damping,
                                damping_bn_err,
                                is_bn);
  }
}

template <typename TensorDataType>
struct inverse_op_gpu
{
  inline __device__ TensorDataType operator()(const TensorDataType& x) const
  {
    // return TensorDataType(1)/x;
    return gpu_lib::pow(x, -1);
  }
};

template <>
void get_matrix_entrywise_inverse(
  El::Matrix<DataType, El::Device::GPU>& input,
  El::Matrix<DataType, El::Device::GPU>& output,
  const El::SyncInfo<El::Device::GPU>& sync_info)
{

  ::lbann::gpu_lib::apply_entrywise_unary_operator<inverse_op_gpu, DataType>(
    input,
    output);
}

template <>
void fill_upper_tri(El::Matrix<DataType, El::Device::GPU>& A,
                    const El::SyncInfo<El::Device::GPU>& sync_info)
{
  const size_t height = A.Height();
  constexpr size_t block_size = 256;
  // TODO: Launch N^2/2 threads instead of N^2
  const size_t grid_size = (height * height + block_size - 1) / block_size;
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(kfac_fill_upper_tri_kernel<DataType>,
                                grid_size,
                                block_size,
                                0,
                                sync_info,
                                A.Buffer(),
                                height);
  }
}

template <>
void update_kronecker_average(El::Matrix<DataType, El::Device::GPU>& Aave,
                              const El::Matrix<DataType, El::Device::GPU>& A,
                              const size_t count,
                              const double decay,
                              const El::SyncInfo<El::Device::GPU>& sync_info)
{
  constexpr size_t block_size = 256;
  const size_t grid_size = (count + block_size - 1) / block_size;
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(kfac_update_kronecker_average_kernel<DataType>,
                                grid_size,
                                block_size,
                                0,
                                sync_info,
                                Aave.Buffer(),
                                A.LockedBuffer(),
                                count,
                                decay);
  }
}

template <>
void identity(El::Matrix<DataType, El::Device::GPU>& A,
              const El::SyncInfo<El::Device::GPU>& sync_info)
{
  const size_t height = A.Height();
  constexpr size_t block_size = 256;
  const size_t num_threads = height * height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(kfac_identity_kernel<DataType>,
                                grid_size,
                                block_size,
                                0,
                                sync_info,
                                A.Buffer(),
                                height);
  }
}

template <>
void pack_lower_tri(El::Matrix<DataType, El::Device::GPU>& L,
                    const El::Matrix<DataType, El::Device::GPU>& A,
                    const El::SyncInfo<El::Device::GPU>& sync_info)
{
  const size_t height = A.Height();
  constexpr size_t block_size = 256;
  const size_t num_threads = height * height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  // TODO: Launch N^2/2 threads instead of N^2
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(kfac_pack_lower_tri_kernel<DataType>,
                                grid_size,
                                block_size,
                                0,
                                sync_info,
                                L.Buffer(),
                                A.LockedBuffer(),
                                height);
  }
}

template <>
void unpack_lower_tri(El::Matrix<DataType, El::Device::GPU>& A,
                      const El::Matrix<DataType, El::Device::GPU>& L,
                      const El::SyncInfo<El::Device::GPU>& sync_info)
{
  const size_t height = A.Height();
  constexpr size_t block_size = 256;
  const size_t num_threads = height * height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  // TODO: Launch N^2/2 threads instead of N^2
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(kfac_unpack_lower_tri_kernel<DataType>,
                                grid_size,
                                block_size,
                                0,
                                sync_info,
                                A.Buffer(),
                                L.LockedBuffer(),
                                height);
  }
}

} // namespace kfac
} // namespace lbann
