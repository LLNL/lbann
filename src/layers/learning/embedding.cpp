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

template <>
void embedding_layer<data_layout::DATA_PARALLEL,El::Device::CPU>::setup_matrices(const El::Grid& grid) {
  Layer::setup_matrices(grid);
  m_gradient_wrt_embeddings.reset(new StarMat<El::Device::CPU>(grid));
}

template <>
void embedding_layer<data_layout::DATA_PARALLEL,El::Device::CPU>::fp_compute() {

  // Local data
  const auto& local_embeddings = dynamic_cast<const CPUMat&>(m_weights[0]->get_values().LockedMatrix());
  const auto& local_input = dynamic_cast<const CPUMat&>(get_local_prev_activations());
  auto& local_output = dynamic_cast<CPUMat&>(get_local_activations());
  const auto& local_width = local_input.Width();

  // Populate output matrix with columns of embedding matrix
  CPUMat embedding_v, output_v;
  for (El::Int col = 0; col < local_width; ++ col) {
    El::View(output_v, local_output, El::ALL, El::IR(col));
    const El::Int ind = static_cast<El::Int>(std::floor(local_input(0, col)));
    if (0 <= ind && ind < static_cast<El::Int>(m_num_embeddings)) {
      El::LockedView(embedding_v, local_embeddings, El::ALL, El::IR(ind));
      El::Copy(embedding_v, output_v);
    } else {
      El::Zero(output_v);
    }
  }

}

template <>
void embedding_layer<data_layout::DATA_PARALLEL,El::Device::CPU>::bp_compute() {

  // Embedding layer is not differentiable w.r.t. inputs
  El::Zero(get_error_signals());

  // Nothing to be done if embeddings are not being optimized
  if (m_weights[0]->get_optimizer() == nullptr) { return; }
  auto& opt = *m_weights[0]->get_optimizer();

  // Local data
  const auto& local_input = dynamic_cast<const CPUMat&>(get_local_prev_activations());
  auto& local_embedding_grad = dynamic_cast<CPUMat&>(m_gradient_wrt_embeddings->Matrix());
  const auto& local_output_grad = dynamic_cast<const CPUMat&>(get_local_prev_error_signals());
  const auto& local_width = local_input.Width();

  // Update appropriate columns of gradient w.r.t. embeddings
  // Note: Don't update gradient for padding index
  El::Zero(local_embedding_grad);
  CPUMat embedding_grad_v, output_grad_v;
  for (El::Int col = 0; col < local_width; ++ col) {
    const El::Int ind = static_cast<El::Int>(std::floor(local_input(0, col)));
    if (0 <= ind
        && ind < static_cast<El::Int>(m_num_embeddings)
        && ind != m_padding_idx) {
      El::View(embedding_grad_v, local_embedding_grad, El::ALL, El::IR(ind));
      El::LockedView(output_grad_v, local_output_grad, El::ALL, El::IR(col));
      El::Axpy(DataType{1}, output_grad_v, embedding_grad_v);
    }
  }
  opt.add_to_gradient(*m_gradient_wrt_embeddings, DataType{1}, true);

}

// Explicit instantiation
template class embedding_layer<data_layout::DATA_PARALLEL, El::Device::CPU>;

} // namespace lbann
