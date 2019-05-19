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

#include "lbann/layers/loss/categorical_accuracy.hpp"
#include "lbann/utils/cuda.hpp"

namespace lbann {

namespace {

/** Fill matrix with corresponding indices.
 *  Indices are equivalent to the global row indices of the input
 *  matrix.
 */
__global__ void fill_indices_kernel(El::Int local_height,
                                    El::Int local_width,
                                    El::Int col_shift,
                                    El::Int col_stride,
                                    El::Int* __restrict__ indices) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int nthreads = blockDim.x * gridDim.x;
  const El::Int size = local_height * local_width;
  for (El::Int pos = gid; pos < size; pos += nthreads) {
    const auto& row = pos % local_height;
    const auto& col = pos / local_height;
    indices[row + col*local_height] = col_shift + row * col_stride;
  }
}

/** Find largest entry within each CUDA block.
 *  Each block is assigned several entries from the same mini-batch
 *  sample and it finds the largest entry. Results are output to
 *  nblocksx x width matrices.
 */
template <El::Int block_size>
__global__ void reduce_max_entries_kernel(El::Int height, El::Int width,
                                          const DataType* __restrict__ values,
                                          El::Int values_row_stride,
                                          El::Int values_col_stride,
                                          const El::Int* __restrict__ indices,
                                          El::Int indices_row_stride,
                                          El::Int indices_col_stride,
                                          DataType* __restrict__ max_values,
                                          El::Int* __restrict__ max_indices) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidx = blockIdx.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nblocksx = gridDim.x;

  // Reduce each matrix column independently
  for (El::Int col = bidy; col < width; col += gridDim.y) {

    // Find largest entry for each thread
    DataType private_max_val = -cuda::infinity<DataType>();
    El::Int private_max_ind = cuda::max<El::Int>();
    for (El::Int row = gidx; row < height; row += nthreadsx) {
      const auto& val = values[row * values_row_stride
                               + col * values_col_stride];
      const auto& ind = indices[row * indices_row_stride
                                + col * indices_col_stride];
      if (val > private_max_val
          || (val == private_max_val && ind < private_max_ind)) {
        private_max_val = val;
        private_max_ind = ind;
      }
    }

    // Shared memory reduction to get largest entry for each block
    __shared__ DataType shared_max_vals[block_size];
    __shared__ El::Int shared_max_inds[block_size];
    shared_max_vals[tid] = private_max_val;
    shared_max_inds[tid] = private_max_ind;
    for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        const auto& val = shared_max_vals[tid + stride];
        const auto& ind = shared_max_inds[tid + stride];
        if (val > shared_max_vals[tid]
          || (val == shared_max_vals[tid] && ind < shared_max_inds[tid])) {
          shared_max_vals[tid] = val;
          shared_max_inds[tid] = ind;
        }
      }
    }
    if (tid == 0) {
      max_values[bidx + col*nblocksx] = shared_max_vals[0];
      max_indices[bidx + col*nblocksx] = shared_max_inds[0];
    }

  }

}

/** Compute sample-wise categorical accuracy.
 *  Outputs one if the prediction and label indices match and
 *  otherwise outputs zero.
 */
__global__ void compute_accuracy_kernel(El::Int local_width,
                                        const El::Int* __restrict__ prediction_indices,
                                        const El::Int* __restrict__ label_indices,
                                        DataType* __restrict__ loss,
                                        El::Int loss_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int nthreads = blockDim.x * gridDim.x;
  constexpr El::Int max_ind = cuda::max<El::Int>();
  for (El::Int col = gid; col < local_width; col += nthreads) {
    const auto& prediction = prediction_indices[col];
    const auto& label = label_indices[col];
    loss[col*loss_ldim] = (prediction == label && prediction < max_ind ?
                           DataType(1) : DataType(0));
  }
}

/** GPU implementation of categorical accuracy layer forward prop. */
void fp_gpu(lbann_comm& comm,
            const AbsDistMat& predictions,
            const AbsDistMat& labels,
            AbsDistMat& loss) {

  // Local matrices
  const auto& local_predictions = predictions.LockedMatrix();
  const auto& local_labels = labels.LockedMatrix();
  auto& local_loss = loss.Matrix();

  // Dimensions
  const auto& height = predictions.Height();
  const auto& local_height = local_predictions.Height();
  const auto& local_width = local_predictions.Width();
  if (local_width < 1) { return; }

  // Column communicator
  auto&& col_comm = predictions.ColComm();
  const auto& col_comm_rank = El::mpi::Rank(col_comm);
  const auto& col_comm_size = El::mpi::Size(col_comm);
  const auto& col_comm_root = loss.RowOwner(0);

  // GPU objects
  auto&& stream = El::GPUManager::Stream();
  auto&& event = El::GPUManager::Event();
  El::SyncInfo<El::Device::GPU> sync_info{stream, event};

  // Initialize CUDA threads/blocks for reduction kernel
  // Note: reduce_max_entries_kernel uses a 2D thread distribution
  // with a 256 x 1 block and nblocksx x local_width grid.
  constexpr El::Int block_size = 256;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.y = local_width;

  // Get indices for all input entries
  cuda::thrust::vector<El::Int> full_inds(local_height * local_width);
  if (full_inds.size() > 0) {
    const El::Int grid_size = (full_inds.size() + block_size - 1) / block_size;
    fill_indices_kernel<<<grid_size, block_size, 0, stream>>>(
      local_height, local_width,
      predictions.ColShift(), predictions.ColStride(),
      full_inds.data().get());
  }

  // Find largest prediction entries in local data
  grid_dims.x = (local_height + block_size - 1) / block_size;
  if (grid_dims.x < 1) { grid_dims.x = 1; }
  cuda::thrust::vector<DataType> prediction_vals(grid_dims.x * local_width);
  cuda::thrust::vector<El::Int> prediction_inds(grid_dims.x * local_width);
  reduce_max_entries_kernel<block_size>
    <<<grid_dims, block_dims, 0, stream>>>(
      local_height, local_width,
      local_predictions.LockedBuffer(), 1, local_predictions.LDim(),
      full_inds.data().get(), 1, local_height,
      prediction_vals.data().get(),
      prediction_inds.data().get());
  while (grid_dims.x > 1) {
    const El::Int prev_height = grid_dims.x;
    grid_dims.x = (prev_height + block_size - 1) / block_size;
    cuda::thrust::vector<DataType> prev_vals(std::move(prediction_vals));
    cuda::thrust::vector<El::Int> prev_inds(std::move(prediction_inds));
    prediction_vals.resize(grid_dims.x * local_width);
    prediction_inds.resize(grid_dims.x * local_width);
    reduce_max_entries_kernel<block_size>
      <<<grid_dims, block_dims, 0, stream>>>(
        prev_height, local_width,
        prev_vals.data().get(), 1, prev_height,
        prev_inds.data().get(), 1, prev_height,
        prediction_vals.data().get(),
        prediction_inds.data().get());
  }

  // Gather large prediction entries
  /// @todo Non-blocking gather
  Al::request prediction_vals_req, prediction_inds_req;
  cuda::thrust::vector<DataType> gathered_prediction_vals;
  cuda::thrust::vector<El::Int> gathered_prediction_inds;
  if (col_comm_size > 1) {
    if (col_comm_rank != col_comm_root) {
      comm.gather(prediction_vals.data().get(), prediction_vals.size(),
                  col_comm_root, col_comm, sync_info);
      comm.gather(prediction_inds.data().get(), prediction_inds.size(),
                  col_comm_root, col_comm, sync_info);
    } else {
      gathered_prediction_vals.resize(prediction_vals.size() * col_comm_size);
      gathered_prediction_inds.resize(prediction_inds.size() * col_comm_size);
      comm.gather(prediction_vals.data().get(), prediction_vals.size(),
                  gathered_prediction_vals.data().get(),
                  col_comm, sync_info);
      comm.gather(prediction_inds.data().get(), prediction_inds.size(),
                  gathered_prediction_inds.data().get(),
                  col_comm, sync_info);
    }
  }

  // Find largest label entries in local data
  grid_dims.x = (local_height + block_size - 1) / block_size;
  if (grid_dims.x < 1) { grid_dims.x = 1; }
  cuda::thrust::vector<DataType> label_vals(grid_dims.x * local_width);
  cuda::thrust::vector<El::Int> label_inds(grid_dims.x * local_width);
  reduce_max_entries_kernel<block_size>
    <<<grid_dims, block_dims, 0, stream>>>(
      local_height, local_width,
      local_labels.LockedBuffer(), 1, local_labels.LDim(),
      full_inds.data().get(), 1, local_height,
      label_vals.data().get(),
      label_inds.data().get());
  while (grid_dims.x > 1) {
    const El::Int prev_height = grid_dims.x;
    grid_dims.x = (prev_height + block_size - 1) / block_size;
    cuda::thrust::vector<DataType> prev_vals(std::move(label_vals));
    cuda::thrust::vector<El::Int> prev_inds(std::move(label_inds));
    label_vals.resize(grid_dims.x * local_width);
    label_inds.resize(grid_dims.x * local_width);
    reduce_max_entries_kernel<block_size>
      <<<grid_dims, block_dims, 0, stream>>>(
        prev_height, local_width,
        prev_vals.data().get(), 1, prev_height,
        prev_inds.data().get(), 1, prev_height,
        label_vals.data().get(),
        label_inds.data().get());
  }

  // Gather large label entries
  /// @todo Non-blocking gather
  Al::request label_vals_req, label_inds_req;
  cuda::thrust::vector<DataType> gathered_label_vals;
  cuda::thrust::vector<El::Int> gathered_label_inds;
  if (col_comm_size > 1) {
    if (col_comm_rank != col_comm_root) {
      comm.gather(label_vals.data().get(), label_vals.size(),
                  col_comm_root, col_comm, sync_info);
      comm.gather(label_inds.data().get(), label_inds.size(),
                  col_comm_root, col_comm, sync_info);
    } else {
      gathered_label_vals.resize(label_vals.size() * col_comm_size);
      gathered_label_inds.resize(label_inds.size() * col_comm_size);
      comm.gather(label_vals.data().get(), label_vals.size(),
                  gathered_label_vals.data().get(),
                  col_comm, sync_info);
      comm.gather(label_inds.data().get(), label_inds.size(),
                  gathered_label_inds.data().get(),
                  col_comm, sync_info);
    }
  }

  // Clean up temporary arrays
  full_inds.clear();

  // Find largest prediction entry in global data
  comm.wait(prediction_vals_req);
  comm.wait(prediction_inds_req);
  if (col_comm_size > 1 && col_comm_rank == col_comm_root) {
    grid_dims.x = (col_comm_size + block_size - 1) / block_size;
    if (grid_dims.x < 1) { grid_dims.x = 1; }
    prediction_vals.resize(grid_dims.x * local_width);
    prediction_inds.resize(grid_dims.x * local_width);
    reduce_max_entries_kernel<block_size>
      <<<grid_dims, block_dims, 0, stream>>>(
        col_comm_size, local_width,
        gathered_prediction_vals.data().get(), col_comm_size, 1,
        gathered_prediction_inds.data().get(), col_comm_size, 1,
        prediction_vals.data().get(),
        prediction_inds.data().get());
    while (grid_dims.x > 1) {
      const El::Int prev_height = grid_dims.x;
      grid_dims.x = (prev_height + block_size - 1) / block_size;
      cuda::thrust::vector<DataType> prev_vals(std::move(prediction_vals));
      cuda::thrust::vector<El::Int> prev_inds(std::move(prediction_inds));
      prediction_vals.resize(grid_dims.x * local_width);
      prediction_inds.resize(grid_dims.x * local_width);
      reduce_max_entries_kernel<block_size>
        <<<grid_dims, block_dims, 0, stream>>>(
          prev_height, local_width,
          prev_vals.data().get(), 1, prev_height,
          prev_inds.data().get(), 1, prev_height,
          prediction_vals.data().get(),
          prediction_inds.data().get());
    }
  }

  // Find largest label entry in global data
  comm.wait(label_vals_req);
  comm.wait(label_inds_req);
  if (col_comm_size > 1 && col_comm_rank == col_comm_root) {
    grid_dims.x = (col_comm_size + block_size - 1) / block_size;
    if (grid_dims.x < 1) { grid_dims.x = 1; }
    label_vals.resize(grid_dims.x * local_width);
    label_inds.resize(grid_dims.x * local_width);
    reduce_max_entries_kernel<block_size>
      <<<grid_dims, block_dims, 0, stream>>>(
        col_comm_size, local_width,
        gathered_label_vals.data().get(), col_comm_size, 1,
        gathered_label_inds.data().get(), col_comm_size, 1,
        label_vals.data().get(),
        label_inds.data().get());
    while (grid_dims.x > 1) {
      const El::Int prev_height = grid_dims.x;
      grid_dims.x = (prev_height + block_size - 1) / block_size;
      cuda::thrust::vector<DataType> prev_vals(std::move(label_vals));
      cuda::thrust::vector<El::Int> prev_inds(std::move(label_inds));
      label_vals.resize(grid_dims.x * local_width);
      label_inds.resize(grid_dims.x * local_width);
      reduce_max_entries_kernel<block_size>
        <<<grid_dims, block_dims, 0, stream>>>(
          prev_height, local_width,
          prev_vals.data().get(), 1, prev_height,
          prev_inds.data().get(), 1, prev_height,
          label_vals.data().get(),
          label_inds.data().get());
    }
  }

  // Compute categorical accuracy
  if (col_comm_rank == col_comm_root) {
    const El::Int grid_size = (local_width + block_size - 1) / block_size;
    compute_accuracy_kernel<<<grid_size, block_size, 0, stream>>>(
      local_width,
      prediction_inds.data().get(), label_inds.data().get(),
      local_loss.Buffer(), local_loss.LDim());
  }

}

} // namespace

template <>
void categorical_accuracy_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(*get_comm(),
         get_prev_activations(0),
         get_prev_activations(1),
         get_activations());
}
template <>
void categorical_accuracy_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(*get_comm(),
         get_prev_activations(0),
         get_prev_activations(1),
         get_activations());
}

} // namespace lbann
