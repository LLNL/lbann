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

#define LBANN_TOP_K_CATEGORICAL_ACCURACY_LAYER_INSTANTIATE
#include "lbann/layers/loss/top_k_categorical_accuracy.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include <thrust/sort.h>
#include <thrust/iterator/discard_iterator.h>

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
      current_entry.value = -cuda::infinity<TensorDataType>();
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

/** Get indices corresponding to one-hot matrix.
 *  Each column of the input matrix is interpreted as a one-hot
 *  vector. Note that we may get race conditions if a matrix column is
 *  not a one-hot vector.
 */
template <typename TensorDataType>
__global__ void one_hot_matrix_to_indices(El::Int local_height,
                                          El::Int local_width,
                                          El::Int global_matrix_col_shift,
                                          El::Int global_matrix_col_stride,
                                          const TensorDataType* __restrict__ local_matrix,
                                          El::Int local_matrix_ldim,
                                          El::Int* __restrict__ indices) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  const El::Int local_size = local_height * local_width;
  for (El::Int i = gid; i < local_size; i += num_threads) {
    const auto& local_row = i % local_height;
    const auto& local_col = i / local_height;
    if (local_matrix[local_row + local_col * local_matrix_ldim] > TensorDataType(0.0)) {
      const auto& global_row = (global_matrix_col_shift
                                + local_row * global_matrix_col_stride);
      indices[local_col] = global_row;
    }
  }
}

/** Compute categorical accuracy for each matrix column.
 *  Loss is one if the label index matches one of the top-k entries
 *  and is otherwise zero.
 */
template <typename TensorDataType>
__global__ void compute_categorical_accuracy(El::Int k,
                                             El::Int width,
                                             El::Int max_entry,
                                             const entry<TensorDataType>*  __restrict__ top_entries,
                                             El::Int top_entries_ldim,
                                             const El::Int*  __restrict__ label_indices,
                                             TensorDataType* __restrict__ loss,
                                             El::Int loss_stride) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  const El::Int num_entries = width * k;
  for (El::Int i = gid; i < num_entries; i += num_threads) {
    const auto& ind = i % k;
    const auto& col = i / k;
    const auto& label_index = label_indices[col];
    if (top_entries[ind + col * top_entries_ldim].index == label_index
        && label_index <= max_entry) {
      loss[col * loss_stride] = TensorDataType(1.0);
    }
  }
}

/** GPU implementation of top-k categorical accuracy layer forward prop. */
template <typename TensorDataType>
void fp_gpu(lbann_comm& comm,
            El::Int k,
            const El::AbstractDistMatrix<TensorDataType>& predictions,
            const El::AbstractDistMatrix<TensorDataType>& labels,
            El::AbstractDistMatrix<TensorDataType>& loss) {

  // Local matrices
  const auto& local_predictions = predictions.LockedMatrix();
  const auto& local_labels = labels.LockedMatrix();
  auto& local_loss = loss.Matrix();
  const auto& height = predictions.Height();
  const auto& local_height = local_predictions.Height();
  const auto& local_width = local_predictions.Width();

  // Trivial cases
  if (k < 1) {
    El::Zero(loss);
    return;
  } else if (k >= height) {
    El::Fill(loss, El::TypeTraits<TensorDataType>::One());
    return;
  } else if (local_width < 1) {
    return;
  }

  // Column communicator
  auto&& col_comm = predictions.ColComm();
  const auto& col_comm_rank = El::mpi::Rank(col_comm);
  const auto& col_comm_size = El::mpi::Size(col_comm);
  const auto& col_comm_root = loss.RowOwner(0);

  // GPU objects
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_loss),
                                     gpu::get_sync_info(local_predictions),
                                     gpu::get_sync_info(local_labels));
  El::SyncInfo<El::Device::GPU> const& sync_info = multisync;
  auto&& stream = sync_info.Stream();
  cuda::thrust::allocator<> alloc(stream);

  // Get label indices
  cuda::thrust::vector<El::Int> label_indices(local_width, height);
  {
    const auto& local_size = local_height * local_width;
    const auto& block_dim = 256;
    const auto& grid_dim = (local_size + block_dim - 1) / block_dim;
    hydrogen::gpu::LaunchKernel(
      one_hot_matrix_to_indices<TensorDataType>,
      grid_dim, block_dim, 0, sync_info,
      local_height, local_width,
      labels.ColShift(), labels.ColStride(),
      local_labels.LockedBuffer(), local_labels.LDim(),
      label_indices.data().get());
    /// @todo The LBANN Aluminum interface doesn't gracefully handle
    /// GPU data that is not TensorDataType.
    El::mpi::AllReduce(label_indices.data().get(),
                       label_indices.size(),
                       El::mpi::MIN,
                       col_comm, sync_info);
  }

  // Find top-k entries in each column of local prediction matrix
  cuda::thrust::vector<entry<TensorDataType>> top_entries(local_width * k);
  {
    const auto& num_local_entries_per_col = std::max(local_height, k);
    const auto& num_local_entries = local_width * num_local_entries_per_col;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_local_entries + block_dim - 1) / block_dim;
    cuda::thrust::vector<entry<TensorDataType>> local_entries(num_local_entries);
    cuda::thrust::vector<El::Int> local_entries_cols(num_local_entries);
    hydrogen::gpu::LaunchKernel(
      dense_matrix_to_sparse_vectors<TensorDataType>,
      grid_dim, block_dim, 0, sync_info,
      num_local_entries_per_col, local_height, local_width, height,
      predictions.ColShift(), predictions.ColStride(),
      local_predictions.LockedBuffer(), local_predictions.LDim(),
      local_entries.data().get(), num_local_entries_per_col);
    hydrogen::gpu::LaunchKernel(
      fill_with_tensor_index,
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
    if (col_comm_rank != col_comm_root) {
      comm.gather(reinterpret_cast<El::byte*>(top_entries.data().get()),
                  top_entries.size() * sizeof(entry<TensorDataType>),
                  col_comm_root,
                  col_comm, sync_info);
    } else {
      cuda::thrust::vector<entry<TensorDataType>> global_top_entries(num_entries);
      cuda::thrust::vector<El::Int> global_top_entries_cols(num_entries);
      comm.gather(reinterpret_cast<El::byte*>(top_entries.data().get()),
                  top_entries.size() * sizeof(entry<TensorDataType>),
                  reinterpret_cast<El::byte*>(global_top_entries.data().get()),
                  col_comm, sync_info);
      hydrogen::gpu::LaunchKernel(
        fill_with_tensor_index,
        grid_dim, block_dim, 0, multisync,
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
        top_entries.data().get(), k,
        global_top_entries.data().get(),
        col_comm_size * k,
        k, local_width,
        sync_info);
    }
  }

  // Compute categorical accuracy
  El::Zero(loss);
  if (col_comm_rank == col_comm_root) {
    const auto& num_entries = local_width * k;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_entries + block_dim - 1) / block_dim;
    hydrogen::gpu::LaunchKernel(
      compute_categorical_accuracy<TensorDataType>,
      grid_dim, block_dim, 0, multisync,
      k, local_width, height-1,
      top_entries.data().get(), k,
      label_indices.data().get(),
      local_loss.Buffer(), local_loss.LDim());
  }

}

} // namespace

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void top_k_categorical_accuracy_layer<TensorDataType, T_layout, Dev>::fp_compute() {
  fp_gpu(*this->get_comm(),
         this->m_k,
         this->get_prev_activations(0),
         this->get_prev_activations(1),
         this->get_activations());
}

#define PROTO(T)                                      \
  template class top_k_categorical_accuracy_layer<    \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;  \
  template class top_k_categorical_accuracy_layer<    \
    T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
