////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#include "lbann/execution_algorithms/kfac/kfac_block_bn.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
__global__ void kfac_compute_bn_factor_data2col_kernel(
  const TensorDataType* __restrict__ activations,
  const TensorDataType* __restrict__ errors,
  const TensorDataType* __restrict__ scales,
  const TensorDataType* __restrict__ biases,
  TensorDataType* __restrict__ cols,
  const size_t batch_size,
  const size_t num_channels,
  const size_t spatial_prod,
  const size_t num_threads)
{ // = batch_size*num_channels*spatial_prod
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < num_threads) {
    const size_t i_c = gid % num_channels;
    const size_t i_n = (gid / num_channels) % batch_size;
    const size_t i_s = gid / num_channels / batch_size;
    const auto scale = scales[i_c];
    const auto bias = biases[i_c];
    const auto i_act =
      i_s + i_c * spatial_prod + i_n * spatial_prod * num_channels;
    const auto error = errors[i_act];
    const auto act = (activations[i_act] - bias) / scale;
    const auto i_out =
      i_c + i_n * num_channels * 2 + i_s * (num_channels * 2 * batch_size);
    cols[i_out] = error * act;
    cols[i_out + num_channels] = error;
  }
}

} // namespace

template <>
void kfac_bn_util::compute_bn_factor_data2col(
  const El::Matrix<DataType, El::Device::GPU>& activations,
  const El::Matrix<DataType, El::Device::GPU>& errors,
  const El::Matrix<DataType, El::Device::GPU>& scales,
  const El::Matrix<DataType, El::Device::GPU>& biases,
  El::Matrix<DataType, El::Device::GPU>& cols,
  const size_t batch_size,
  const size_t num_channels,
  const size_t spatial_prod,
  const El::SyncInfo<El::Device::GPU>& sync_info)
{
  constexpr size_t block_size = 256;
  const size_t num_threads = batch_size * num_channels * spatial_prod;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(
      kfac_compute_bn_factor_data2col_kernel<DataType>,
      grid_size,
      block_size,
      0,
      sync_info,
      activations.LockedBuffer(),
      errors.LockedBuffer(),
      scales.LockedBuffer(),
      biases.LockedBuffer(),
      cols.Buffer(),
      batch_size,
      num_channels,
      spatial_prod,
      num_threads);
  }
}

} // namespace lbann
