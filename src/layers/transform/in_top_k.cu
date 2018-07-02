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

#include "lbann/layers/transform/in_top_k.hpp"
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

/** Initialize sparse vector from local input data. */
__global__ void initialize_entries(El::Int k,
                                   El::Int num_entries,
                                   El::Int local_height,
                                   El::Int local_width,
                                   El::Int col_shift,
                                   El::Int col_stride,
                                   const DataType* __restrict__ local_input,
                                   El::Int local_input_ldim,
                                   entry*  __restrict__ entries,
                                   El::Int* __restrict__ cols) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  const El::Int num_entries_per_col = num_entries / local_width;
  for (El::Int i = gid; i < num_entries; i += num_threads) {
    const auto& row = i % num_entries_per_col;
    const auto& col = i / num_entries_per_col;
    if (row < local_height) {
      const auto& global_row = col_shift + row * col_stride;
      entries[i].value = local_input[row + col * local_input_ldim];
      entries[i].index = global_row;
    } else {
      entries[i].value = entry::min_value;
      entries[i].index = entry::max_index;
    }
    cols[i] = col;
  }  
}

/** Fill an array with tensor dimension indices.
 *  Entries in 'indices' are populated with the dimension index for a
 *  corresponding entry in a packed tensor.
 */
__global__ void fill_tensor_indices(El::Int tensor_size,
                                    El::Int dim_stride,
                                    El::Int dim_max,
                                    El::Int* indices) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int i = gid; i < tensor_size; i += num_threads) {
    indices[i] = (i / dim_stride) % dim_max;
  }  
}

/** Set selected entries in local output data to one. */
__global__ void indicate_entries(El::Int k,
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
                                 const entry*  __restrict__ entries) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int i = gid; i < num_entries; i += num_threads) {
    const auto& col = i / k;
    const auto& global_row = entries[i].index;
    const auto& row_owner = (global_row + col_align) % col_stride;
    if (global_row < height && row_owner == col_rank) {
      const auto& row = (global_row > col_shift ?
                         (global_row - col_shift - 1) / col_stride + 1 :
                         0);
      local_output[row + col * local_output_ldim] = DataType(1);
    }
  }  
}

/** CPU implementation of in_top_k layer forward prop. */
void fp_gpu(lbann_comm& comm,
            El::Int k, const AbsDistMat& input, AbsDistMat& output) {
  if (input.Wrap() != El::ELEMENT || output.Wrap() != El::ELEMENT) {
    LBANN_ERROR("in_top_k layer GPU implementation assumes elemental "
                "distributed matrices");
  }

  // Trivial cases
  if (k < 1) {
    El::Zero(output);
    return;
  } else if (k >= input.Height()) {
    El::Fill(output, DataType(1));
    return;
  }
  
  // Local matrices
  const auto& local_input = input.LockedMatrix();
  auto& local_output = output.Matrix();
  const El::Int height = input.Height();
  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();

  // Initialize GPU objects
  using entry_array = El::Memory<entry, El::Device::GPU>;
  using entry_ptr = thrust::device_ptr<entry>;
  using index_array = El::Memory<El::Int, El::Device::GPU>;
  using index_ptr = thrust::device_ptr<El::Int>;
#ifdef HYDROGEN_HAVE_CUB
  const unsigned int memory_mode = 1; // CUB GPU memory pool
#else
  const unsigned int memory_mode = 0;
#endif // HYDROGEN_HAVE_CUB
  auto stream = El::GPUManager::Stream();

  // Find top-k entries in each local matrix column
  entry_array top_k_entries(k * local_width, memory_mode);
  {
    const auto& num_entries_per_col = std::max(local_height, k);
    const auto& num_entries = num_entries_per_col * local_width;
    entry_array entries(num_entries, memory_mode);
    index_array cols(num_entries, memory_mode);
    const auto& block_dim = 256;
    const auto& grid_dim = (num_entries + block_dim - 1) / block_dim;
    initialize_entries<<<grid_dim, block_dim, 0, stream>>>(
      k, num_entries, local_height, local_width,
      input.ColShift(), input.ColStride(),
      local_input.LockedBuffer(), local_input.LDim(),
      entries.Buffer(), cols.Buffer());
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        entry_ptr(entries.Buffer()),
                        entry_ptr(entries.Buffer() + num_entries),
                        index_ptr(cols.Buffer()),
                        entry_compare());
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                               index_ptr(cols.Buffer()),
                               index_ptr(cols.Buffer() + num_entries),
                               entry_ptr(entries.Buffer()));
    CHECK_CUDA(cudaMemcpy2DAsync(top_k_entries.Buffer(),
                                 k * sizeof(entry),
                                 entries.Buffer(),
                                 num_entries_per_col * sizeof(entry),
                                 k * sizeof(entry),
                                 local_width,
                                 cudaMemcpyDeviceToDevice,
                                 stream));
  }

  // Find top-k entries in each global matrix column
  // Note: Nothing needs to be done if matrix columns are not
  // distributed.
  auto&& col_comm = input.ColComm();
  const El::Int col_comm_size = El::mpi::Size(col_comm);
  if (col_comm_size > 1) {
    const auto& num_entries_per_rank = k * local_width;
    const auto& num_entries = num_entries_per_rank * col_comm_size;
    entry_array entries(num_entries, memory_mode);
    index_array cols(num_entries, memory_mode);
    comm.all_gather(reinterpret_cast<El::byte*>(top_k_entries.Buffer()),
                    num_entries_per_rank * sizeof(entry) / sizeof(El::byte),
                    reinterpret_cast<El::byte*>(entries.Buffer()),
                    num_entries_per_rank * sizeof(entry) / sizeof(El::byte),
                    col_comm);
    const auto& block_dim = 256;
    const auto& grid_dim = (num_entries + block_dim - 1) / block_dim;
    fill_tensor_indices<<<grid_dim, block_dim, 0, stream>>>(
      num_entries, k, local_width, cols.Buffer());
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        entry_ptr(entries.Buffer()),
                        entry_ptr(entries.Buffer() + num_entries),
                        index_ptr(cols.Buffer()),
                        entry_compare());
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                               index_ptr(cols.Buffer()),
                               index_ptr(cols.Buffer() + num_entries),
                               entry_ptr(entries.Buffer()));
    CHECK_CUDA(cudaMemcpy2DAsync(top_k_entries.Buffer(),
                                 k * sizeof(entry),
                                 entries.Buffer(),
                                 k * col_comm_size * sizeof(entry),
                                 k * sizeof(entry),
                                 local_width,
                                 cudaMemcpyDeviceToDevice,
                                 stream));
  }

  // Indicate output entries corresponding to top-k input entries
  El::Zero(output);
  if (output.Participating() && local_height > 0 && local_width > 0) {
    const auto& num_entries = k * local_width;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_entries + block_dim - 1) / block_dim;
    indicate_entries<<<grid_dim, block_dim, 0, stream>>>(
      k, num_entries, height, local_height, local_width,
      output.ColRank(), output.ColAlign(),
      output.ColShift(), output.ColStride(),
      local_output.Buffer(), local_output.LDim(),
      top_k_entries.Buffer());
  }

}

} // namespace

template <>
void in_top_k_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(*get_comm(), m_k, get_prev_activations(), get_activations());
}
template <>
void in_top_k_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(*get_comm(), m_k, get_prev_activations(), get_activations());
}

} // namespace lbann
