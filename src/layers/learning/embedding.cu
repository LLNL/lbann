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

#define LBANN_EMBEDDING_LAYER_INSTANTIATE
#include "lbann/layers/learning/embedding.hpp"

namespace lbann {

namespace {

/** @brief Kernel for forward prop
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (embedding_dim / bsize) x mini_batch_size x 1
 */
template <typename TensorDataType>
__global__ void fp_kernel(El::Int num_embeddings,
                          El::Int embedding_dim,
                          El::Int mini_batch_size,
                          const TensorDataType* __restrict__ indices,
                          El::Int indices_stride,
                          const TensorDataType* __restrict__ embeddings,
                          El::Int embeddings_ldim,
                          TensorDataType* __restrict__ output,
                          El::Int output_ldim) {
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nthreadsy = blockDim.y * gridDim.y;
  for (El::Int j = gidy; j < mini_batch_size; j += nthreadsy) {
    const El::Int ind = static_cast<El::Int>(indices[j*indices_stride]);
    for (El::Int i = gidx; i < embedding_dim; i += nthreadsx) {
      auto& y = output[i+j*output_ldim];
      if (0 <= ind && ind < num_embeddings) {
        y = embeddings[i+ind*embeddings_ldim];
      }
      else {
        y = TensorDataType{0};
      }
    }
  }
}

/** @brief Kernel for backprop
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (embedding_dim / bsize) x mini_batch_size x 1
 */
template <typename TensorDataType>
__global__ void bp_kernel(El::Int num_embeddings,
                          El::Int embedding_dim,
                          El::Int mini_batch_size,
                          El::Int padding_idx,
                          const TensorDataType* __restrict__ indices,
                          El::Int indices_stride,
                          const TensorDataType* __restrict__ gradient_wrt_output,
                          El::Int gradient_wrt_output_ldim,
                          TensorDataType* __restrict__ gradient_wrt_embeddings,
                          El::Int gradient_wrt_embeddings_ldim) {
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nthreadsy = blockDim.y * gridDim.y;
  for (El::Int j = gidy; j < mini_batch_size; j += nthreadsy) {
    const El::Int ind = static_cast<El::Int>(indices[j*indices_stride]);
    if (0 <= ind && ind < num_embeddings && ind != padding_idx) {
      for (El::Int i = gidx; i < embedding_dim; i += nthreadsx) {
        const auto& dy = gradient_wrt_output[i+j*gradient_wrt_output_ldim];
        auto& dw = gradient_wrt_embeddings[i+ind*gradient_wrt_embeddings_ldim];
        cuda::atomic_add(&dw, dy);
      }
    }
  }
}

} // namespace

template <typename TensorDataType>
void setup_matrices_impl(embedding_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::GPU>& l, const El::Grid& grid) {
  l.m_gradient_wrt_embeddings.reset(new El::DistMatrix<TensorDataType, El::STAR, El::STAR, El::ELEMENT, El::Device::GPU>(grid));
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::setup_matrices(const El::Grid& grid) {
  data_type_layer<TensorDataType>::setup_matrices(grid);
  setup_matrices_impl<TensorDataType>(*this, grid);
}

template <typename TensorDataType>
void fp_compute_impl(embedding_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::GPU>& l) {

  // Local data
  const auto& local_embeddings = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(l.get_data_type_weights()[0]->get_values().LockedMatrix());
  const auto& local_input = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(l.get_local_prev_activations());
  auto& local_output = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(l.get_local_activations());

  // Launch CUDA kernel
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_output.Height() + block_size - 1) / block_size;
    grid_dims.y = local_output.Width();
    fp_kernel<TensorDataType><<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
      m_num_embeddings,
      m_embedding_dim,
      local_input.Width(),
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_embeddings.LockedBuffer(),
      local_embeddings.LDim(),
      local_output.Buffer(),
      local_output.LDim());
  }

}

template <typename TensorDataType>
void bp_compute_impl(embedding_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::GPU>& l) {

  // Embedding layer is not differentiable w.r.t. inputs
  El::Zero(get_error_signals());

  // Nothing to be done if embeddings are not being optimized
  if (get_data_type_weights()[0]->get_optimizer() == nullptr) { return; }
  auto& opt = *get_data_type__weights()[0]->get_optimizer();

  // Local data
  const auto& local_input = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(l.get_local_prev_activations());
  auto& local_embedding_grad = dynamic_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(l.m_gradient_wrt_embeddings->Matrix());
  const auto& local_output_grad = dynamic_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(l.get_local_prev_error_signals());

  // Launch CUDA kernel
  El::Zero(local_embedding_grad);
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_output_grad.Height() + block_size - 1) / block_size;
    grid_dims.y = local_output_grad.Width();
    bp_kernel<TensorDataType><<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
      m_num_embeddings,
      m_embedding_dim,
      local_input.Width(),
      m_padding_idx,
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_output_grad.LockedBuffer(),
      local_output_grad.LDim(),
      local_embedding_grad.Buffer(),
      local_embedding_grad.LDim());
  }
  opt.add_to_gradient(*l.m_gradient_wrt_embeddings, TensorDataType{1}, true);

}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::fp_compute() {
  fp_compute_impl<TensorDataType>(*this);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::bp_compute() {
  bp_compute_impl<TensorDataType>(*this);
}

// Explicit instantiation
template class embedding_layer<float, data_layout::DATA_PARALLEL, El::Device::GPU>;

} // namespace lbann
