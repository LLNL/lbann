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

#define LBANN_ONE_HOT_LAYER_INSTANTIATE
#include "lbann/layers/misc/one_hot_impl.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

/** See El::AbstractDistMatrix::RowOwner. */
__device__ __forceinline__ El::Int
distmat_index_owner(El::Int global_index, El::Int align, El::Int stride)
{
  return (global_index + align) % stride;
}

/** See El::AbstractDistMatrix::LocalRow. */
__device__ __forceinline__ El::Int distmat_local_index(El::Int global_index,
                                                       El::Int rank,
                                                       El::Int align,
                                                       El::Int stride)
{
  auto shift = (rank - align) % stride;
  if (global_index > shift) {
    return (global_index - shift - 1) / stride + 1;
  }
  else {
    return 0;
  }
}

/**
 *  On input, output is assumed to be filled with zeros.
 *
 *  Block dimensions: bdim x 1 x 1
 *
 *  Grid dimensions: (local_mini_batch_size / bdim) x 1 x 1
 */
template <typename TensorDataType>
__global__ void fp_kernel(El::Int local_mini_batch_size,
                          El::Int output_size,
                          El::Int col_rank,
                          const TensorDataType* __restrict__ local_input,
                          El::Int input_ldim,
                          TensorDataType* __restrict__ local_output,
                          El::Int output_ldim,
                          El::Int output_col_align,
                          El::Int output_col_stride)
{
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int nthreads = blockDim.x * gridDim.x;
  for (El::Int j = gid; j < local_mini_batch_size; j += nthreads) {
    const auto& x = local_input[j * input_ldim];
    const auto i_global = static_cast<El::Int>(gpu_lib::floor(x));
    const auto owner_rank =
      distmat_index_owner(i_global, output_col_align, output_col_stride);
    if (0 <= i_global && i_global < output_size && owner_rank == col_rank) {
      const auto i = distmat_local_index(i_global,
                                         col_rank,
                                         output_col_align,
                                         output_col_stride);
      local_output[i + j * output_ldim] = TensorDataType(1.f);
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void one_hot_layer<TensorDataType, Layout, Device>::fp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("one_hot_layer::fp_compute");

  // Local matrices
  using AbsLocalMat = El::AbstractMatrix<TensorDataType>;
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& input = this->get_prev_activations();
  auto& output = this->get_activations();
  auto& local_output = dynamic_cast<LocalMat&>(output.Matrix());
  const El::Int local_mini_batch_size = output.LocalWidth();
  const El::Int output_size = output.Height();

  // Make sure all procs in column communicator have access to input
  LocalMat local_input;
  const auto& col_comm = input.ColComm();
  const auto col_rank = El::mpi::Rank(col_comm);
  const auto owner_rank = input.RowOwner(0);
  if (col_rank == owner_rank) {
    El::LockedView(local_input, input.LockedMatrix());
  }
  else {
    local_input.Resize(1, input.LocalWidth());
  }
  /** @todo (tym1 3/12/21): We are working around a bug in Hydrogen.
   *  Broadcast with Matrix<T,D> is not instatiated. */
  El::Broadcast(static_cast<El::AbstractMatrix<TensorDataType>&>(local_input),
                col_comm,
                owner_rank);

  // Populate one-hot vectors
  El::Zero(output);
  if (!local_output.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_input),
                                       gpu::get_sync_info(local_output));
    constexpr size_t block_size = 64;
    const size_t grid_size =
      (local_mini_batch_size + block_size - 1) / block_size;
    hydrogen::gpu::LaunchKernel(fp_kernel<TensorDataType>,
                                grid_size,
                                block_size,
                                0,
                                multisync,
                                local_mini_batch_size,
                                output_size,
                                col_rank,
                                local_input.LockedBuffer(),
                                local_input.LDim(),
                                output.Buffer(),
                                output.LDim(),
                                output.ColAlign(),
                                output.ColStride());
  }
}

#define PROTO(T)                                                               \
  template class one_hot_layer<T,                                              \
                               data_layout::DATA_PARALLEL,                     \
                               El::Device::GPU>;                               \
  template class one_hot_layer<T, data_layout::MODEL_PARALLEL, El::Device::GPU>
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
