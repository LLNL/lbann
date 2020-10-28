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

#define LBANN_IN_TOP_K_LAYER_INSTANTIATE
#include "lbann/layers/transform/in_top_k.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace lbann {

namespace {

/** Sparse vector entry. */
template <typename TensorDataType>
struct entry {
  /** Vector entry value. */
  TensorDataType value;
  /** Vector entry index. */
  El::Int index;
};

/** Comparison operation to sort sparse vector entries.
 *  Entries are sorted by value in decreasing order, with ties broken
 *  in favor of entries with smaller indices.
 */
template <typename TensorDataType>
struct entry_compare : ::thrust::binary_function<entry<TensorDataType>,entry<TensorDataType>,bool> {
  __host__ __device__ bool operator()(const entry<TensorDataType>& a, const entry<TensorDataType>& b) const {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
  }
};

/** Convert columns of a dense matrix into sparse vectors.
 *  The matrix and vectors are both distributed, so entry indices in
 *  the sparse vectors correspond to global row indices in the dense
 *  matrix.
 */
template <typename TensorDataType>
__global__ void dense_matrix_to_sparse_vectors(El::Int local_vector_size,
                                               El::Int local_matrix_height,
                                               El::Int local_matrix_width,
                                               El::Int global_matrix_height,
                                               El::Int global_matrix_col_shift,
                                               El::Int global_matrix_col_stride,
                                               const TensorDataType* __restrict__ local_matrix,
                                               El::Int local_matrix_ldim,
                                               entry<TensorDataType>* __restrict__ local_entries,
                                               El::Int local_entries_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  const El::Int num_local_entries = local_vector_size * local_matrix_width;
  for (El::Int i = gid; i < num_local_entries; i += num_threads) {
    const auto& local_row = i % local_vector_size;
    const auto& local_col = i / local_vector_size;
    auto& current_entry = local_entries[local_row + local_col * local_entries_ldim];
    if (local_row < local_matrix_height) {
      const auto& global_row = (global_matrix_col_shift
                                + local_row * global_matrix_col_stride);
      current_entry.value = local_matrix[local_row + local_col * local_matrix_ldim];
      current_entry.index = global_row;
    } else {
      current_entry.value = -gpu_lib::infinity<DataType>();
      current_entry.index = global_matrix_height;
    }
  }
}

/** Fill an array with a corresponding tensor index.
 *  Consider a d(1) x d(2) x ... x d(n) tensor with entry indices
 *  denoted with (i(1), ..., i(n)). This tensor is contiguous in
 *  memory with d(1) as the most major dimension and d(n) as the most
 *  minor (e.g. d(1) is the width and d(2) is the height for a
 *  column-major matrix). Given some k, this kernel sets each entry in
 *  the tensor to i(k). Using this notation:
 *    tensor_size = d(1) * ... * d(n)
 *    dim         = d(k)
 *    dim_stride  = d(k+1) * ... * d(n)
 */
template <typename TensorDataType>
__global__ void fill_with_tensor_index(El::Int tensor_size,
                                       El::Int dim,
                                       El::Int dim_stride,
                                       El::Int* tensor) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int i = gid; i < tensor_size; i += num_threads) {
    tensor[i] = (i / dim_stride) % dim;
  }
}

/** Set selected matrix entries to one.
 *  Each matrix column corresponds to k entries in 'entries'. If a
 *  local matrix entry corresponds to one of the top-k entries, then
 *  it is set to one.
 */
template <typename TensorDataType>
__global__ void indicate_matrix_entries(El::Int k,
                                        El::Int global_matrix_height,
                                        El::Int local_matrix_height,
                                        El::Int local_matrix_width,
                                        El::Int global_matrix_col_rank,
                                        El::Int global_matrix_col_align,
                                        El::Int global_matrix_col_shift,
                                        El::Int global_matrix_col_stride,
                                        TensorDataType* __restrict__ local_matrix,
                                        El::Int local_matrix_ldim,
                                        const entry<TensorDataType>*  __restrict__ entries,
                                        El::Int entries_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  const El::Int num_entries = local_matrix_width * k;
  for (El::Int i = gid; i < num_entries; i += num_threads) {
    const auto& ind = i % k;
    const auto& local_col = i / k;
    const auto& global_row = entries[ind + local_col * entries_ldim].index;
    const auto& row_owner = ((global_row + global_matrix_col_align)
                             % global_matrix_col_stride);
    if (global_row < global_matrix_height
        && row_owner == global_matrix_col_rank) {
      El::Int local_row = 0;
      if (global_row > global_matrix_col_shift) {
        local_row = ((global_row - global_matrix_col_shift - 1)
                     / global_matrix_col_stride + 1);
      }
      local_matrix[local_row + local_col * local_matrix_ldim] = TensorDataType(1.0);
    }
  }
}

/** GPU implementation of in_top_k layer forward prop. */
template <typename TensorDataType>
void fp_gpu(lbann_comm& comm,
            El::Int k,
            const El::AbstractDistMatrix<TensorDataType>& input,
            El::AbstractDistMatrix<TensorDataType>& output) {
  if (input.Wrap() != El::ELEMENT || output.Wrap() != El::ELEMENT) {
    LBANN_ERROR("in_top_k layer GPU implementation assumes elemental "
                "distributed matrices");
  }

  // Local matrices
  const auto& local_input = input.LockedMatrix();
  auto& local_output = output.Matrix();
  const auto& height = input.Height();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // Trivial cases
  if (k < 1) {
    El::Zero(output);
    return;
  } else if (k >= height) {
    El::Fill(output, El::TypeTraits<TensorDataType>::One());
    return;
  } else if (local_width < 1) {
    return;
  }

  // Column communicator
  auto&& col_comm = input.ColComm();
  const auto& col_comm_rank = El::mpi::Rank(col_comm);
  const auto& col_comm_size = El::mpi::Size(col_comm);

  // GPU objects
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_output),
                                     gpu::get_sync_info(local_input));
  El::SyncInfo<El::Device::GPU> sync_info = multisync;
  gpu_lib::thrust::allocator<> alloc(sync_info.Stream());

  // Find top-k entries in each column of local prediction matrix
  gpu_lib::thrust::vector<entry<TensorDataType>> top_entries(local_width * k);
  {
    const auto& num_local_entries_per_col = std::max(local_height, k);
    const auto& num_local_entries = local_width * num_local_entries_per_col;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_local_entries + block_dim - 1) / block_dim;
    gpu_lib::thrust::vector<entry<TensorDataType>> local_entries(num_local_entries);
    gpu_lib::thrust::vector<El::Int> local_entries_cols(num_local_entries);
    hydrogen::gpu::LaunchKernel(
      dense_matrix_to_sparse_vectors<TensorDataType>,
      grid_dim, block_dim, 0, sync_info,
      num_local_entries_per_col, local_height, local_width, height,
      input.ColShift(), input.ColStride(),
      local_input.LockedBuffer(), local_input.LDim(),
      local_entries.data().get(), num_local_entries_per_col);
    hydrogen::gpu::LaunchKernel(
      fill_with_tensor_index<TensorDataType>,
      grid_dim, block_dim, 0, sync_info,
      num_local_entries, local_width, num_local_entries_per_col,
      local_entries_cols.data().get());
    ::thrust::sort_by_key(alloc.system(),
                          local_entries.begin(),
                          local_entries.end(),
                          local_entries_cols.begin(),
                          entry_compare<TensorDataType>());
    ::thrust::stable_sort_by_key(alloc.system(),
                                 local_entries_cols.begin(),
                                 local_entries_cols.end(),
                                 local_entries.begin());
    hydrogen::gpu::Copy2DIntraDevice(
      local_entries.data().get(), num_local_entries_per_col,
      top_entries.data().get(), k,
      k, local_width,
      sync_info);
  }

  // Find top-k entries in each column of global prediction matrix
  if (col_comm_size > 1) {
    const auto& num_entries_per_rank = local_width * k;
    const auto& num_entries = col_comm_size * num_entries_per_rank;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_entries + block_dim - 1) / block_dim;
    gpu_lib::thrust::vector<entry<TensorDataType>> global_top_entries(num_entries);
    gpu_lib::thrust::vector<El::Int> global_top_entries_cols(num_entries);
    comm.all_gather(reinterpret_cast<El::byte*>(top_entries.data().get()),
                    top_entries.size() * sizeof(entry<TensorDataType>),
                    reinterpret_cast<El::byte*>(global_top_entries.data().get()),
                    top_entries.size() * sizeof(entry<TensorDataType>),
                    col_comm, sync_info);
    hydrogen::gpu::LaunchKernel(
      fill_with_tensor_index<TensorDataType>,
      grid_dim, block_dim, 0, sync_info,
      num_entries, local_width, k, global_top_entries_cols.data().get());
    ::thrust::sort_by_key(alloc.system(),
                          global_top_entries.begin(),
                          global_top_entries.end(),
                          global_top_entries_cols.begin(),
                          entry_compare<TensorDataType>());
    ::thrust::stable_sort_by_key(alloc.system(),
                                 global_top_entries_cols.begin(),
                                 global_top_entries_cols.end(),
                                 global_top_entries.begin());
    hydrogen::gpu::Copy2DIntraDevice(
      global_top_entries.data().get(), col_comm_size * k,
      top_entries.data().get(), k,
      k, local_width,
      sync_info);
  }

  // Indicate output entries corresponding to top-k input entries
  El::Zero(output);
  if (output.Participating() && local_height > 0 && local_width > 0) {
    const auto& num_entries = local_width * k;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_entries + block_dim - 1) / block_dim;
    hydrogen::gpu::LaunchKernel(
      indicate_matrix_entries<TensorDataType>,
      grid_dim, block_dim, 0, sync_info,
      k, height, local_height, local_width,
      output.ColRank(), output.ColAlign(),
      output.ColShift(), output.ColStride(),
      local_output.Buffer(), local_output.LDim(),
      top_entries.data().get(), k);
  }

}

} // namespace

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void in_top_k_layer<TensorDataType, T_layout, Dev>::fp_compute() {
  fp_gpu(*this->get_comm(),
         this->m_k,
         this->get_prev_activations(),
         this->get_activations());
}

#define PROTO(T)                                     \
  template class in_top_k_layer<                     \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class in_top_k_layer<                     \
    T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
