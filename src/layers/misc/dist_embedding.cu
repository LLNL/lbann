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

#include "lbann/layers/misc/dist_embedding.hpp"
#ifdef LBANN_HAS_NVSHMEM

#include "lbann/utils/gpu/helpers.hpp"
#include "lbann/utils/nvshmem.hpp"

namespace lbann
{
namespace
{

// Typedefs
using Size2 = cuda::array<size_t, 2>;
template <typename T>
using VectorMetadata = typename dist_embedding_layer<T,data_layout::DATA_PARALLEL,El::Device::GPU>::vector_metadata;

/** Copy between two device buffers, using all threads in a warp. */
template <typename T> __device__ __forceinline__
T* memcpy_warp(T* __restrict__ dest, const T* __restrict__ src, size_t n) {
  constexpr size_t warp_size = 32;
  for (size_t i = threadIdx.x; i < n; i += warp_size) {
    dest[i] = src[i];
  }
  __syncwarp();
  return dest;
}

/** See El::AbstractDistMatrix::ColOwner. */
__device__ __forceinline__
size_t distmat_index_owner(size_t global_index, size_t align, size_t stride) {
  return (global_index + align) % stride;
}

/** See El::AbstractDistMatrix::GlobalCol. */
__device__ __forceinline__
size_t distmat_global_index(size_t local_index, size_t shift, size_t stride) {
  return shift + local_index * stride;
}

/** See El::AbstractDistMatrix::LocalCol. */
__device__ __forceinline__
size_t distmat_local_index(size_t global_index, size_t rank, size_t align, size_t stride) {
  auto shift = (stride + rank - align) % stride;
  if (global_index > shift) {
    return (global_index - shift - 1) / stride + 1;
  }
  else {
    return 0;
  }
}

/** Launch a CUDA kernel.
 *
 *  @todo Check that argument types match kernel signature.
 */
template <typename Kernel, typename... Args>
inline void launch_cuda_kernel(
  const Kernel& kernel,
  dim3 grid_dims,
  dim3 block_dims,
  size_t shared_mem,
  cudaStream_t stream,
  Args... args) {
  void* arg_list[] = {
    const_cast<void*>(reinterpret_cast<const void*>(&args))...
  };
  CHECK_CUDA(
    cudaLaunchKernel(
      reinterpret_cast<const void*>(&kernel),
      grid_dims,
      block_dims,
      arg_list,
      shared_mem,
      stream));
}

/** Launch a collective NVSHMEM kernel.
 *
 *  Needed for device-side NVSHMEM synchronization calls like
 *  nvshmem_wait. If grid_dims is zero, then the NVSHMEM will launch
 *  with the largest available grid.
 *
 *  @todo Check that argument types match kernel signature.
 */
template <typename Kernel, typename... Args>
inline void launch_nvshmem_collective_kernel(
  const Kernel& kernel,
  dim3 grid_dims,
  dim3 block_dims,
  size_t shared_mem,
  cudaStream_t stream,
  Args... args) {
  if (grid_dims.x == 0) {
    grid_dims.y = 0;
    grid_dims.z = 0;
  }
  void* arg_list[] = {
    const_cast<void*>(reinterpret_cast<const void*>(&args))...
  };
  auto status = nvshmemx_collective_launch(
    reinterpret_cast<const void*>(&kernel),
    grid_dims,
    block_dims,
    arg_list,
    shared_mem,
    stream);
  if (status != 0) {
    LBANN_ERROR(
      "Failed to launch NVSHMEM collective kernel ",
      "(error ",status,")");
  }
}

} // namespace <anon>

// ---------------------------------------------
// Life cycle and setup
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>::~dist_embedding_layer()
{
  if (m_embeddings_buffer != nullptr) {
    nvshmem_free(m_embeddings_buffer);
  }
  if (m_workspace_buffer != nullptr) {
    nvshmem_free(m_workspace_buffer);
  }
  if (m_metadata_buffer != nullptr) {
    nvshmem_free(m_metadata_buffer);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::attach_embeddings_to_shmem_buffer() {
  if (m_embeddings_buffer != nullptr || m_embeddings_buffer_size != 0) {
    LBANN_ERROR("attempted to attach embedding matrix ",
                "to NVSHMEM buffer multiple times");
  }

  // Embedding weights matrix
  using ValuesGetter = weights_details::SafeWeightsAccessor<TensorDataType>;
  auto& embeddings = ValuesGetter::mutable_values(this->get_weights(0));
  const auto dist = embeddings.DistData();
  if (dist.device != El::Device::GPU) {
    LBANN_ERROR("attempted to attach non-GPU matrix to NVSHMEM buffer");
  }
#if 0 // nvshmem_addr_accessible is not supported as of NVSHMEM 1.4
  if (nvshmem_addr_accessible(embeddings.LockedBuffer(), nvshmem_my_pe())) {
    return;
  }
#endif

  // Calculate size of NVSHMEM buffer
  const auto col_comm_size = El::mpi::Size(embeddings.ColComm());
  const auto row_comm_size = El::mpi::Size(embeddings.RowComm());
  const auto height = embeddings.Height();
  const auto width = embeddings.Width();
  const auto local_height = (height + col_comm_size - 1) / col_comm_size;
  const auto local_width = (width + row_comm_size - 1) / row_comm_size;
  m_embeddings_buffer_size = local_height * local_width * sizeof(TensorDataType);
  if (m_embeddings_buffer_size == 0) {
    return;
  }

  // Allocate NVSHMEM buffer
  m_embeddings_buffer = nvshmem::malloc<TensorDataType>(m_embeddings_buffer_size);

  // Attach matrix to NVSHMEM buffer
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> orig_mat(
    embeddings.Construct(embeddings.Grid(), embeddings.Root()));
  *orig_mat = std::move(embeddings);
  embeddings.Empty();
  embeddings.AlignWith(dist);
  dynamic_cast<El::ElementalMatrix<TensorDataType>&>(embeddings).Attach(
    height, width,
    *dist.grid, dist.colAlign, dist.rowAlign,
    m_embeddings_buffer, local_height, dist.root);
  El::Copy(*orig_mat, embeddings);

}

// ---------------------------------------------
// Forward prop
// ---------------------------------------------

namespace
{

/** Request embedding vectors from owner processes.
 *
 *  Block dimensions: 32 x 1 x 1
 *
 *  Grid dimensions: input_dims[1] x input_dims[0] x 1
 */
template <typename T>
__global__ void request_embeddings_kernel(
  size_t embedding_dim,
  Size2 input_dims,
  const T* __restrict__ input,
  Size2 input_strides,
  const T* __restrict__ embeddings,
  Size2 embeddings_strides,
  VectorMetadata<T>* __restrict__ metadata,
  Size2 metadata_strides,
  T* __restrict__ workspace,
  Size2 workspace_strides,
  size_t rank,
  size_t input_rowshift,
  size_t input_rowstride,
  size_t embeddings_rowalign,
  size_t embeddings_rowstride) {

  // Indices
  const size_t bidx = blockIdx.x;
  const size_t bidy = blockIdx.y;
  const size_t nblocksx = gridDim.x;
  const size_t nblocksy = gridDim.y;

  const size_t i_per_block = (input_dims[1] + nblocksx - 1) / nblocksx;
  const size_t i_start = bidx * i_per_block;
  const size_t i_end = cuda::min((bidx+1) * i_per_block, input_dims[1]);
  for (size_t j = bidy; j < input_dims[0]; j += nblocksy) {
    for (size_t i = i_start; i < i_end; ++i) {
      const auto& global_j = distmat_global_index(j, input_rowshift, input_rowstride);

      // Get embedding vector index
      const auto& global_index_float = input[i*input_strides[1] + j*input_strides[0]];
      const auto& global_index = static_cast<size_t>(cuda::floor(global_index_float));

      // Figure out which process owns embedding vector
      __shared__ unsigned char metadata_shared[sizeof(VectorMetadata<T>)];
      auto& m = *reinterpret_cast<VectorMetadata<T>*>(metadata_shared);
      if (threadIdx.x == 0) {
        m.source_rank = distmat_index_owner(global_index, embeddings_rowalign, embeddings_rowstride);
        m.source_index = distmat_local_index(global_index, m.source_rank, embeddings_rowalign, embeddings_rowstride);
        m.target_rank = rank;
        m.target_index = i + global_j*input_dims[1];
        m.is_active = true;
        metadata[i*metadata_strides[1] + global_j*metadata_strides[0]] = m;
      }
      __syncwarp();

      // Get embedding vector from owner process
      nvshmemx_getmem_nbi_warp(
        &workspace[m.target_index * workspace_strides[0]],
        &embeddings[m.source_index * embeddings_strides[0]],
        embedding_dim*sizeof(T),
        m.source_rank);

    }
  }

}

/** Copy embedding vectors to output tensor.
 *
 *  Block dimensions: 32 x 1 x 1
 *
 *  Grid dimensions: input_dims[1] x input_dims[0] x 1
 */
template <typename T>
__global__ void copy_embeddings_kernel(
  size_t embedding_dim,
  Size2 input_dims,
  const VectorMetadata<T>* __restrict__ metadata,
  Size2 metadata_strides,
  const T* __restrict__ workspace,
  Size2 workspace_strides,
  T* __restrict__ output,
  Size2 output_strides,
  size_t input_rowshift,
  size_t input_rowstride) {

  // Indices
  const size_t bidx = blockIdx.x;
  const size_t bidy = blockIdx.y;
  const size_t nblocksx = gridDim.x;
  const size_t nblocksy = gridDim.y;

  const size_t i_per_block = (input_dims[1] + nblocksx - 1) / nblocksx;
  const size_t i_start = bidx * i_per_block;
  const size_t i_end = cuda::min((bidx+1) * i_per_block, input_dims[1]);
  for (size_t j = bidy; j < input_dims[0]; j += nblocksy) {
    for (size_t i = i_start; i < i_end; ++i) {
      const auto& global_j = distmat_global_index(j, input_rowshift, input_rowstride);
      const auto& m = metadata[i*metadata_strides[1] + global_j*metadata_strides[0]];
      memcpy_warp(
        &output[i*embedding_dim + j*output_strides[0]],
        &workspace[m.target_index * workspace_strides[0]],
        embedding_dim);
    }
  }

}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::fp_compute() {

  // Data matrices
  // Note: Make sure to get original weight values since they are in
  // SHMEM buffer.
  using ValuesGetter = weights_details::SafeWeightsAccessor<TensorDataType>;
  const auto& embeddings = ValuesGetter::mutable_values(this->get_weights(0));
  const auto& input = this->get_prev_activations();
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<LocalMat&>(this->get_local_activations());

  // Dimensions
  const size_t input_size = this->get_input_size();
  const size_t output_size = this->get_output_size();
  const size_t mini_batch_size = input.Width();
  const size_t local_mini_batch_size = local_input.Width();

  // GPU objects
  auto&& stream = hydrogen::cuda::GetDefaultStream();
  nvshmem::initialize();

  // Barrier to handle gradient checking
  /// @todo Think of a way to avoid this synchronization
  if (m_barrier_in_forward_prop) {
    nvshmemx_barrier_all_on_stream(stream);
  }

  // Synchronize non-blocking barrier
  // Note: Make sure embeddings are up-to-date and NVSHMEM workspaces
  // are safe to reset.
  auto& comm = *this->get_comm();
  comm.wait(m_nb_barrier_request);

  // Initialize NVSHMEM buffer for communicating embedding vectors
  if (m_workspace_buffer_size < output_size * mini_batch_size) {
    m_workspace_buffer_size = output_size * mini_batch_size;
    m_workspace_buffer = nvshmem::realloc(m_workspace_buffer,
                                          m_workspace_buffer_size);
  }
  LocalMat workspace(
    m_embedding_dim,
    input_size * mini_batch_size,
    m_workspace_buffer,
    m_embedding_dim);

  // Initialize NVSHMEM buffer for embedding vector metadata
  if (m_metadata_buffer_size < input_size * mini_batch_size) {
    m_metadata_buffer_size = input_size * mini_batch_size;
    m_metadata_buffer = nvshmem::realloc(m_metadata_buffer,
                                         m_metadata_buffer_size);
  }
  CHECK_CUDA(
    cudaMemsetAsync(
      m_metadata_buffer,
      0,
      m_metadata_buffer_size*sizeof(vector_metadata),
      stream));

  // Request embedding vectors from owning processes
  const size_t rank = comm.get_rank_in_trainer();
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size = 32;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = input_size;
    grid_dims.y = local_mini_batch_size;
    launch_cuda_kernel(
      request_embeddings_kernel<TensorDataType>,
      grid_dims,
      block_dims,
      0,
      stream,
      m_embedding_dim,
      Size2{local_mini_batch_size, input_size},
      local_input.LockedBuffer(),
      Size2{size_t(local_input.LDim()), 1},
      embeddings.LockedBuffer(),
      Size2{size_t(embeddings.LDim()), 1},
      m_metadata_buffer,
      Size2{input_size, 1},
      workspace.Buffer(),
      Size2{size_t(workspace.LDim()), 1},
      size_t(rank),
      size_t(input.RowShift()),
      size_t(input.RowStride()),
      size_t(embeddings.RowAlign()),
      size_t(embeddings.RowStride()));
  }
  nvshmemx_quiet_on_stream(stream);

  // Copy embedding vectors to output tensor
  if (!local_output.IsEmpty()) {
    constexpr size_t block_size = 32;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = input_size;
    grid_dims.y = local_mini_batch_size;
    launch_cuda_kernel(
      copy_embeddings_kernel<TensorDataType>,
      grid_dims,
      block_dims,
      0,
      stream,
      m_embedding_dim,
      Size2{local_mini_batch_size, input_size},
      m_metadata_buffer,
      Size2{input_size, 1},
      workspace.LockedBuffer(),
      Size2{size_t(workspace.LDim()), 1},
      local_output.Buffer(),
      Size2{size_t(local_output.LDim()), 1},
      size_t(input.RowShift()),
      size_t(input.RowStride()));
  }

  // Non-blocking barrier
  // Note: NVSHMEM workspaces are ready to recieve gradients.
  nb_barrier(comm, comm.get_trainer_comm(), m_nb_barrier_request);

}

// ---------------------------------------------
// Backprop
// ---------------------------------------------

namespace
{

/** Send gradients to owner processes.
 *
 *  Block dimensions: 32 x 1 x 1
 *
 *  Grid dimensions: input_dims[1] x input_dims[0] x 1
 */
template <typename T>
__global__ void send_gradients_kernel(
  size_t embedding_dim,
  Size2 input_dims,
  const T* __restrict__ output_grad,
  Size2 output_grad_strides,
  VectorMetadata<T>* __restrict__ metadata,
  Size2 metadata_strides,
  T* __restrict__ workspace,
  Size2 workspace_strides,
  size_t input_rowshift,
  size_t input_rowstride) {

  // Indices
  const size_t bidx = blockIdx.x;
  const size_t bidy = blockIdx.y;
  const size_t nblocksx = gridDim.x;
  const size_t nblocksy = gridDim.y;

  // Assign metadata to CUDA blocks
  const size_t i_per_block = (input_dims[1] + nblocksx - 1) / nblocksx;
  const size_t i_start = bidx * i_per_block;
  const size_t i_end = cuda::min((bidx+1) * i_per_block, input_dims[1]);

  // Send gradients to owner processes
  for (size_t j = bidy; j < input_dims[0]; j += nblocksy) {
    for (size_t i = i_start; i < i_end; ++i) {
      const auto& global_j = distmat_global_index(j, input_rowshift, input_rowstride);
      auto& m = metadata[i*metadata_strides[1] + global_j*metadata_strides[0]];
      auto* workspace_ptr = &workspace[m.target_index * workspace_strides[0]];
      memcpy_warp(
        workspace_ptr,
        &output_grad[i*embedding_dim + j*output_grad_strides[0]],
        embedding_dim);
      if (m.source_rank != m.target_rank) {
        nvshmemx_putmem_nbi_warp(
          workspace_ptr,
          workspace_ptr,
          embedding_dim*sizeof(T),
          m.source_rank);
        nvshmemx_putmem_nbi_warp(
          &m,
          &m,
          sizeof(VectorMetadata<T>),
          m.source_rank);
      }
    }
  }

}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::bp_compute() {

  // Data matrices
  const auto& input = this->get_prev_activations();
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(this->get_local_prev_error_signals());

  // Dimensions
  const size_t input_size = this->get_input_size();
  const size_t mini_batch_size = input.Width();
  const size_t local_mini_batch_size = local_output_grad.Width();

  // GPU objects
  auto&& stream = hydrogen::cuda::GetDefaultStream();

  // Synchronize non-blocking barrier
  // Note: Make sure NVSHMEM workspaces are ready to recieve gradients.
  auto& comm = *this->get_comm();
  comm.wait(m_nb_barrier_request);

  // Initialize NVSHMEM buffer for gradient w.r.t. embeddings
  LocalMat workspace(
    m_embedding_dim,
    input_size * mini_batch_size,
    m_workspace_buffer,
    m_embedding_dim);

  // Send gradients to owner processes
  if (!local_output_grad.IsEmpty()) {
    constexpr size_t block_size = 32;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = input_size;
    grid_dims.y = local_mini_batch_size;
    launch_cuda_kernel(
      send_gradients_kernel<TensorDataType>,
      grid_dims,
      block_dims,
      0,
      stream,
      m_embedding_dim,
      Size2{local_mini_batch_size, input_size},
      local_output_grad.LockedBuffer(),
      Size2{size_t(local_output_grad.LDim()), 1},
      m_metadata_buffer,
      Size2{input_size, 1},
      workspace.Buffer(),
      Size2{size_t(workspace.LDim()), 1},
      size_t(input.RowShift()),
      size_t(input.RowStride()));
  }
  nvshmemx_quiet_on_stream(stream);

  // Non-blocking barrier
  // Note: Gradients have been sent.
  nb_barrier(comm, comm.get_trainer_comm(), m_nb_barrier_request);

  // Use dense optimizer if needed
  if (!m_sparse_sgd) {

    // Create buffer for dense gradients
    const auto& embeddings = this->weights_values(0);
    std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> embeddings_grad(
      embeddings.Construct(embeddings.Grid(), embeddings.Root()));
    embeddings_grad->AlignWith(embeddings);
    El::Zeros(*embeddings_grad, embeddings.Height(), embeddings.Width());
    auto& local_embeddings_grad = dynamic_cast<LocalMat&>(embeddings_grad->Matrix());

    // Apply SGD step to convert sparse gradients to dense gradients
    apply_sparse_sgd_step(
      input_size * mini_batch_size,
      local_embeddings_grad);

    // Send dense gradients to dense optimizer
    auto* opt = this->get_weights(0).get_optimizer();
    if (opt != nullptr) {
      opt->add_to_gradient(*embeddings_grad);
    }

  }

}

// ---------------------------------------------
// Sparse SGD
// ---------------------------------------------

namespace
{

/** Sparse SGD on local embeddings.
 *
 *  Block dimensions: 32 x 1 x 1
 *
 *  Grid dimensions: num_gradients x 1 x 1
 */
template <typename T>
__global__ void sgd_kernel(
  T learning_rate,
  size_t embedding_dim,
  size_t num_gradients,
  const VectorMetadata<T>* __restrict__ metadata,
  const T* __restrict__ embeddings_grad,
  Size2 embeddings_grad_strides,
  T* __restrict__ embeddings,
  Size2 embeddings_strides,
  size_t rank) {

  // Indices
  const size_t tid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t nblocks = gridDim.x;
  constexpr size_t warp_size = 32;

  // Assign requests to CUDA blocks
  const size_t gradients_per_block = (num_gradients + nblocks - 1) / nblocks;
  const size_t i_start = bid * gradients_per_block;
  const size_t i_end = cuda::min((bid+1) * gradients_per_block, num_gradients);

  for (size_t i = i_start; i < i_end; ++i) {
    const auto& m = metadata[i];
    if (m.is_active && m.source_rank == rank) {

      // Update embedding vector with gradient
      const auto* __restrict__ dw = &embeddings_grad[m.target_index * embeddings_grad_strides[0]];
      auto* __restrict__ w = &embeddings[m.source_index * embeddings_strides[0]];
      for (size_t k = tid; k < embedding_dim; k += warp_size) {
        cuda::atomic_add(&w[k], -learning_rate * dw[k]);
      }

    }
  }

}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::apply_sparse_sgd_step(
  size_t num_gradients,
  LocalMat& local_embeddings) {

  // GPU objects
  auto&& stream = hydrogen::cuda::GetDefaultStream();

  // Synchronize non-blocking barrier
  // Note: Make sure gradients have been received.
  auto& comm = *this->get_comm();
  comm.wait(m_nb_barrier_request);

  // Initialize SHMEM buffer for gradient w.r.t. embeddings
  LocalMat local_embeddings_grad(
    m_embedding_dim,
    num_gradients,
    m_workspace_buffer,
    m_embedding_dim);

  // Sparse SGD on local embeddings
  const size_t rank = comm.get_rank_in_trainer();
  constexpr size_t block_size = 32;
  const size_t grid_size = num_gradients;
  launch_cuda_kernel(
    sgd_kernel<TensorDataType>,
    grid_size,
    block_size,
    0,
    stream,
    m_learning_rate,
    m_embedding_dim,
    num_gradients,
    m_metadata_buffer,
    local_embeddings_grad.LockedBuffer(),
    Size2{size_t(local_embeddings_grad.LDim()), 1},
    local_embeddings.Buffer(),
    Size2{size_t(local_embeddings.LDim()), 1},
    rank);

}

// ---------------------------------------------
// Explicit template instantiation
// ---------------------------------------------

/// @todo fp16
template class dist_embedding_layer<
  float, data_layout::DATA_PARALLEL, El::Device::GPU>;
template class dist_embedding_layer<
  double, data_layout::DATA_PARALLEL, El::Device::GPU>;

} // namespace lbann
#endif // LBANN_HAS_NVSHMEM
