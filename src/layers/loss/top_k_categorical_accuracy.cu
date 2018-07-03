////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/loss/top_k_categorical_accuracy.hpp"
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/exception.hpp"

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace lbann {

namespace {

/** Sparse vector entry. */
struct entry {

  /** Vector entry value. */
  DataType value;
  /** Vector entry index. */
  El::Int index;

  /** Minimum possible value. */
  static constexpr DataType min_value = -std::numeric_limits<DataType>::infinity();
  /** Maximum possible index. */
  static constexpr El::Int max_index = std::numeric_limits<El::Int>::max();

};

/** Comparison operation to sort sparse vector entries.
 *  Entries are sorted by value in decreasing order. Entries with the
 *  same value are sorted by index in increasing order.
 */
struct entry_compare : thrust::binary_function<entry,entry,bool> {
  __host__ __device__ bool operator()(const entry& a, const entry& b) const {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
  }
};

/** Initialize sparse vector entries from local input data.
 *  Each column of the input matrix is converted into a sparse
 *  vector.
 */
__global__ void initialize_local_entries(El::Int num_local_entries,
                                         El::Int local_height,
                                         El::Int local_width,
                                         El::Int col_shift,
                                         El::Int col_stride,
                                         const DataType* __restrict__ local_input,
                                         El::Int local_input_ldim,
                                         entry*  __restrict__ local_entries,
                                         El::Int* __restrict__ local_entry_cols) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  const El::Int num_local_entries_per_col = num_local_entries / local_width;
  for (El::Int i = gid; i < num_local_entries; i += num_threads) {
    const auto& row = i % num_local_entries_per_col;
    const auto& col = i / num_local_entries_per_col;
    if (row < local_height) {
      const auto& global_row = col_shift + row * col_stride;
      local_entries[i].value = local_input[row + col * local_input_ldim];
      local_entries[i].index = global_row;
    } else {
      local_entries[i].value = entry::min_value;
      local_entries[i].index = entry::max_index;
    }
    local_entry_cols[i] = col;
  }  
}

/** Fill an array with tensor dimension indices.
 *  Entries in 'indices' are populated with the dimension index for a
 *  corresponding entry in a packed tensor.
 */
__global__ void fill_tensor_indices(El::Int tensor_size,
                                    El::Int dim_max,
                                    El::Int dim_stride,
                                    El::Int* indices) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int i = gid; i < tensor_size; i += num_threads) {
    indices[i] = (i / dim_stride) % dim_max;
  }  
}

/** Set selected entries in local output data to one. */
__global__ void indicate_local_entries(El::Int k,
                                       El::Int num_entries,
                                       El::Int height,
                                       El::Int local_height,
                                       El::Int local_width,
                                       El::Int col_rank,
                                       El::Int col_align,
                                       El::Int col_shift,
                                       El::Int col_stride,
                                       DataType* __restrict__ local_output,
                                       El::Int local_output_ldim,
                                       const entry*  __restrict__ entries,
                                       El::Int entries_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int i = gid; i < num_entries; i += num_threads) {
    const auto& ind = i % k;
    const auto& col = i / k;
    const auto& global_row = entries[ind + col * entries_ldim].index;
    const auto& row_owner = (global_row + col_align) % col_stride;
    if (global_row < height && row_owner == col_rank) {
      const auto& row = (global_row > col_shift ?
                         (global_row - col_shift - 1) / col_stride + 1 :
                         0);
      local_output[row + col * local_output_ldim] = DataType(1);
    }
  }  
}

/** GPU implementation of top-k categorical accuracy layer forward prop. */
void fp_gpu(lbann_comm& comm,
            El::Int k, const AbsDistMat& input, AbsDistMat& output) {
  if (input.Wrap() != El::ELEMENT || output.Wrap() != El::ELEMENT) {
    LBANN_ERROR("top-k categorical accuracy layer GPU implementation assumes elemental "
                "distributed matrices");
  }

  // Local matrices
  const auto& local_input = input.LockedMatrix();
  auto& local_output = output.Matrix();
  const El::Int height = input.Height();
  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();

  // Trivial cases
  if (k < 1) {
    El::Zero(output);
    return;
  } else if (k >= height) {
    El::Fill(output, DataType(1));
    return;
  } else if (local_width < 1) {
    return;
  }
  
  // Column communicator
  auto&& col_comm = input.ColComm();
  const auto& col_comm_rank = El::mpi::Rank(col_comm);
  const auto& col_comm_size = El::mpi::Size(col_comm);

  // GPU objects
  using entry_array = El::Memory<entry, El::Device::GPU>;
  using entry_ptr = thrust::device_ptr<entry>;
  using index_array = El::Memory<El::Int, El::Device::GPU>;
  using index_ptr = thrust::device_ptr<El::Int>;
#ifdef HYDROGEN_HAVE_CUB
  const unsigned int memory_mode = 1; // CUB GPU memory pool
#else
  const unsigned int memory_mode = 0;
#endif // HYDROGEN_HAVE_CUB
  auto&& stream = El::GPUManager::Stream();
  entry_array top_entries(col_comm_size * local_width * k, memory_mode);
  auto* local_top_entries = (top_entries.Buffer()
                             + col_comm_rank * local_width * k);

  // Find top-k entries in each local matrix column
  {
    const auto& num_local_entries_per_col = std::max(local_height, k);
    const auto& num_local_entries = local_width * num_local_entries_per_col;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_local_entries + block_dim - 1) / block_dim;
    entry_array local_entries(num_local_entries, memory_mode);
    index_array local_entry_cols(num_local_entries, memory_mode);
    initialize_local_entries<<<grid_dim, block_dim, 0, stream>>>(
      num_local_entries, local_height, local_width,
      input.ColShift(), input.ColStride(),
      local_input.LockedBuffer(), local_input.LDim(),
      local_entries.Buffer(), local_entry_cols.Buffer());
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        entry_ptr(local_entries.Buffer()),
                        entry_ptr(local_entries.Buffer() + num_local_entries),
                        index_ptr(local_entry_cols.Buffer()),
                        entry_compare());
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                               index_ptr(local_entry_cols.Buffer()),
                               index_ptr(local_entry_cols.Buffer() + num_local_entries),
                               entry_ptr(local_entries.Buffer()));
    CHECK_CUDA(cudaMemcpy2DAsync(local_top_entries,
                                 k * sizeof(entry),
                                 local_entries.Buffer(),
                                 num_local_entries_per_col * sizeof(entry),
                                 k * sizeof(entry),
                                 local_width,
                                 cudaMemcpyDeviceToDevice,
                                 stream));
  }

  // Find top-k entries in each global matrix column
  if (col_comm_size > 1) {
    const auto& num_entries_per_rank = local_width * k;
    const auto& num_entries = col_comm_size * num_entries_per_rank;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_entries + block_dim - 1) / block_dim;
    comm.all_gather(reinterpret_cast<El::byte*>(MPI_IN_PLACE),
                    num_entries_per_rank * sizeof(entry),
                    reinterpret_cast<El::byte*>(top_entries.Buffer()),
                    num_entries_per_rank * sizeof(entry),
                    col_comm);
    index_array top_entry_cols(num_entries, memory_mode);
    fill_tensor_indices<<<grid_dim, block_dim, 0, stream>>>(
      num_entries, local_width, k, top_entry_cols.Buffer());
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        entry_ptr(top_entries.Buffer()),
                        entry_ptr(top_entries.Buffer() + num_entries),
                        index_ptr(top_entry_cols.Buffer()),
                        entry_compare());
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                               index_ptr(top_entry_cols.Buffer()),
                               index_ptr(top_entry_cols.Buffer() + num_entries),
                               entry_ptr(top_entries.Buffer()));
  }

  // Indicate output entries corresponding to top-k input entries
  El::Zero(output);
  if (output.Participating() && local_height > 0 && local_width > 0) {
    const auto& num_entries = local_width * k;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_entries + block_dim - 1) / block_dim;
    indicate_local_entries<<<grid_dim, block_dim, 0, stream>>>(
      k, num_entries, height, local_height, local_width,
      output.ColRank(), output.ColAlign(),
      output.ColShift(), output.ColStride(),
      local_output.Buffer(), local_output.LDim(),
      top_entries.Buffer(), col_comm_size * k);
  }

}

} // namespace

template <>
void top_k_categorical_accuracy_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  LBANN_ERROR("not yet implemented");
}
template <>
void top_k_categorical_accuracy_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  LBANN_ERROR("not yet implemented");
}

} // namespace lbann
