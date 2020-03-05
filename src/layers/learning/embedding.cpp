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

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::setup_matrices(const El::Grid& grid) {
  data_type_layer<TensorDataType>::setup_matrices(grid);
  this->m_gradient_wrt_embeddings.reset(new El::DistMatrix<TensorDataType, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>(grid));
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::fp_compute() {

  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Local data
  const auto& local_embeddings = dynamic_cast<const CPUMatType&>(this->get_data_type_weights(0).get_values().LockedMatrix());
  const auto& local_input = dynamic_cast<const CPUMatType&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<CPUMatType&>(this->get_local_activations());
  const auto& local_width = local_input.Width();

  // Populate output matrix with columns of embedding matrix
  CPUMatType embedding_v, output_v;
  for (El::Int col = 0; col < local_width; ++ col) {
    El::View(output_v, local_output, El::ALL, El::IR(col));
    const El::Int ind = static_cast<El::Int>(std::floor(local_input(0, col)));
    if (0 <= ind && ind < static_cast<El::Int>(this->m_num_embeddings)) {
      El::LockedView(embedding_v, local_embeddings, El::ALL, El::IR(ind));
      El::Copy(embedding_v, output_v);
    } else {
      El::Zero(output_v);
    }
  }

}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::bp_compute() {
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Embedding layer is not differentiable w.r.t. inputs
  El::Zero(this->get_error_signals());

  // Nothing to be done if embeddings are not being optimized
  if (this->get_data_type_weights(0).get_optimizer() == nullptr) { return; }
  auto& opt = *this->get_data_type_weights(0).get_optimizer();

  // Local data
  const auto& local_input = dynamic_cast<const CPUMatType&>(this->get_local_prev_activations());
  auto& local_embedding_grad = dynamic_cast<CPUMatType&>(this->m_gradient_wrt_embeddings->Matrix());
  const auto& local_output_grad = dynamic_cast<const CPUMatType&>(this->get_local_prev_error_signals());
  const auto& local_width = local_input.Width();

  // Update appropriate columns of gradient w.r.t. embeddings
  // Note: Don't update gradient for padding index
  El::Zero(local_embedding_grad);
  CPUMatType embedding_grad_v, output_grad_v;
  for (El::Int col = 0; col < local_width; ++ col) {
    const El::Int ind = static_cast<El::Int>(std::floor(local_input(0, col)));
    if (0 <= ind
        && ind < static_cast<El::Int>(this->m_num_embeddings)
        && ind != this->m_padding_idx) {
      El::View(embedding_grad_v, local_embedding_grad, El::ALL, El::IR(ind));
      El::LockedView(output_grad_v, local_output_grad, El::ALL, El::IR(col));
      El::Axpy(DataType{1}, output_grad_v, embedding_grad_v);
    }
  }
  opt.add_to_gradient(*this->m_gradient_wrt_embeddings, TensorDataType{1}, true);

}

// Explicit instantiation
template class embedding_layer<DataType, data_layout::DATA_PARALLEL, El::Device::CPU>;

} // namespace lbann
