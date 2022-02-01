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
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/execution_algorithms/kfac/kfac_block_fc_conv.hpp"


namespace lbann {

namespace {

template <typename TensorDataType>
__global__ void kfac_get_diagonal_kernel(
    TensorDataType * __restrict__ diag,
    const TensorDataType * __restrict__ A,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height)
    diag[gid] = A[gid+gid*height];
}

template <typename TensorDataType>
__global__ void kfac_conv_transpose_kernel(
    const TensorDataType * __restrict__ A,
    TensorDataType * __restrict__ Acol,
    const size_t num_channels,
    const size_t spatial_prod, const size_t num_elems) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < num_elems) {
    const auto i_spatial = gid%spatial_prod;
    const auto i_c = (gid/spatial_prod)%num_channels;
    const auto i_n = (gid/spatial_prod/num_channels);
    Acol[i_c+i_spatial*num_channels+i_n*num_channels*spatial_prod] = A[gid];
  }
}

} // namespace

template <>
void kfac_fc_conv_util::get_diagonal<El::Device::GPU>(
    El::Matrix<DataType, El::Device::GPU>& diag,
    const El::Matrix<DataType, El::Device::GPU>& A,
    const El::SyncInfo<El::Device::GPU>& sync_info) {
  const size_t height = A.Height();
  constexpr size_t block_size = 256;
  const size_t num_threads = height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  hydrogen::gpu::LaunchKernel(
    kfac_get_diagonal_kernel<DataType>,
    grid_size, block_size, 0, sync_info,
    diag.Buffer(), A.LockedBuffer(), height);
  /*
  kfac_get_diagonal_kernel<DataType>
      <<<grid_size, block_size, 0, sync_info.Stream()>>>(
          diag.Buffer(), A.LockedBuffer(), height);
          */
}

template <>
void kfac_fc_conv_util::conv_transpose<El::Device::GPU>(
    const El::Matrix<DataType, El::Device::GPU>& activations,
    El::Matrix<DataType, El::Device::GPU>& act_columns,
    size_t mini_batch_size, size_t num_channels,
    size_t spatial_prod,
    const El::SyncInfo<El::Device::GPU>& sync_info) {
  constexpr size_t block_size = 256;
  const size_t num_elems = mini_batch_size*num_channels*spatial_prod;
  const size_t grid_size = (num_elems + block_size - 1) / block_size;
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(
      kfac_conv_transpose_kernel<DataType>,
      grid_size, block_size, 0, sync_info,
      activations.LockedBuffer(), act_columns.Buffer(),
      num_channels, spatial_prod,
      num_elems);
  }
}

} // namespace lbann
