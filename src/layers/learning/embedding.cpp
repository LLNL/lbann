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

#define LBANN_EMBEDDING_LAYER_INSTANTIATE
#include "lbann/layers/learning/embedding.hpp"
#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void embedding_layer<TensorDataType, Layout, Device>::fp_compute()
{
  using MatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Local data
  const auto& local_embeddings =
    dynamic_cast<const MatType&>(this->weights_values(0).LockedMatrix());
  const auto& local_input =
    dynamic_cast<const MatType&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<MatType&>(this->get_local_activations());
  const size_t input_size = this->get_input_size();
  const size_t local_mini_batch_size = local_input.Width();

  // Populate output matrix with values from embedding matrix
  MatType embedding_v, output_v;
  for (size_t j = 0; j < local_mini_batch_size; ++j) {
    for (size_t i = 0; i < input_size; ++i) {
      El::View(output_v,
               local_output,
               El::IR(i * m_embedding_dim, (i + 1) * m_embedding_dim),
               El::IR(j));
      const El::Int ind = static_cast<El::Int>(std::floor(local_input(i, j)));
      if (0 <= ind && ind < static_cast<El::Int>(this->m_num_embeddings)) {
        El::LockedView(embedding_v, local_embeddings, El::ALL, El::IR(ind));
        El::Copy(embedding_v, output_v);
      }
      else {
        El::Zero(output_v);
      }
    }
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void embedding_layer<TensorDataType, Layout, Device>::bp_compute()
{
  using MatType = El::Matrix<TensorDataType, El::Device::CPU>;
  const TensorDataType one = El::TypeTraits<TensorDataType>::One();

  // Embedding layer is not differentiable w.r.t. inputs
  El::Zero(this->get_error_signals());

  // Nothing to be done if embeddings are not being optimized
  if (this->get_weights(0).get_optimizer() == nullptr) {
    return;
  }
  auto& opt = *this->get_weights(0).get_optimizer();

  // Local data
  const auto& local_input =
    dynamic_cast<const MatType&>(this->get_local_prev_activations());
  const auto& local_output_grad =
    dynamic_cast<const MatType&>(this->get_local_prev_error_signals());

  TensorDataType dst_scale, gradient_scale;
  auto& embeddings_grad =
    opt.get_gradient_buffer(dst_scale, gradient_scale, true);
  auto& local_embedding_grad = dynamic_cast<MatType&>(embeddings_grad.Matrix());

  const size_t input_size = this->get_input_size();
  const size_t local_mini_batch_size = local_input.Width();

  // Update gradient w.r.t. embeddings
  // Note: Don't update gradient for padding index
  if (dst_scale == El::TypeTraits<TensorDataType>::Zero()) {
    El::Zero(local_embedding_grad);
  }
  else if (dst_scale != one) {
    El::Scale(dst_scale, local_embedding_grad);
  }

  MatType embedding_grad_v, output_grad_v;
  for (size_t j = 0; j < local_mini_batch_size; ++j) {
    for (size_t i = 0; i < input_size; ++i) {
      const El::Int ind = static_cast<El::Int>(std::floor(local_input(i, j)));
      if (0 <= ind && ind < static_cast<El::Int>(this->m_num_embeddings) &&
          ind != this->m_padding_idx) {
        El::LockedView(output_grad_v,
                       local_output_grad,
                       El::IR(i * m_embedding_dim, (i + 1) * m_embedding_dim),
                       El::IR(j));
        El::View(embedding_grad_v, local_embedding_grad, El::ALL, El::IR(ind));
        El::Axpy(gradient_scale, output_grad_v, embedding_grad_v);
      }
    }
  }
}

// Explicit instantiation
#define PROTO(T)                                                               \
  template class embedding_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
