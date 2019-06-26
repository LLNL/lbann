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

#include "lbann/layers/learning/embedding.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

template <>
void embedding_layer<data_layout::DATA_PARALLEL,El::Device::CPU>::setup_matrices(const El::Grid& grid) {
  Layer::setup_matrices(grid);
  m_dictionary_gradient = StarMat<El::Device::CPU>(grid);
}

template <>
void embedding_layer<data_layout::DATA_PARALLEL,El::Device::CPU>::setup_dims() {
  Layer::setup_dims();

  // Make sure input dimensions are valid
  if (this->get_input_size() != 1) {
    const auto& input_dims = this->get_input_dims();
    std::ostringstream err;
    err << get_type() << " layer \"" << get_name() << "\" "
        << "recieved an input tensor with invalid dimensions "
        << "(expected 1, got ";
    for (size_t i = 0; i < input_dims.size(); ++i) {
      err << (i > 0 ? "x" : "") << input_dims[i];
    }
    err << ")";
    LBANN_ERROR(err.str());
  }

  // Output is size of embedding vector
  this->set_output_dims({static_cast<int>(m_embedding_size)});

}

template <>
void embedding_layer<data_layout::DATA_PARALLEL,El::Device::CPU>::setup_data() {
  Layer::setup_data();

  // Make sure layer has weights for dictionary
  if (this->m_weights.size() != 1) {
    std::ostringstream err;
    err << "attempted to setup "
        << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "with an invalid number of weights "
        << "(expected 1, "
        << "found " << this->m_weights.size() << ")";
    LBANN_ERROR(err.str());
  }

  // Initialize dictionary
  auto& dict = *m_weights[0];
  auto matrix_dist = get_prev_activations().DistData();
  matrix_dist.colDist = El::STAR;
  matrix_dist.rowDist = El::STAR;
  dict.set_dims({static_cast<int>(m_embedding_size)},
                {static_cast<int>(m_dictionary_size)});
  dict.set_matrix_distribution(matrix_dist);

  // Initialize gradient w.r.t. dictionary
  m_dictionary_gradient.Resize(m_embedding_size, m_dictionary_size);

}

template <>
void embedding_layer<data_layout::DATA_PARALLEL,El::Device::CPU>::fp_compute() {

  // Local data
  const auto& local_dict = m_weights[0]->get_values().LockedMatrix();
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  const auto& local_width = local_input.Width();

  // Populate output matrix with appropriate columns of dictionary
  CPUMat dict_v, output_v;
  for (El::Int col = 0; col < local_width; ++ col) {
    const El::Int ind = static_cast<El::Int>(local_input(0, col));
    El::LockedView(dict_v, local_dict, El::ALL, El::IR(ind));
    El::View(output_v, local_output, El::ALL, El::IR(col));
    El::Copy(dict_v, output_v);
  }

}

template <>
void embedding_layer<data_layout::DATA_PARALLEL,El::Device::CPU>::bp_compute() {

  // Embedding layer is not differentiable w.r.t. inputs
  El::Zero(get_error_signals());

  // Nothing to be done if dictionary is not being optimized
  if (m_weights[0]->get_optimizer() == nullptr) { return; }
  auto& opt = *m_weights[0]->get_optimizer();

  // Local data
  const auto& local_input = get_local_prev_activations();
  auto& local_dict_grad = m_dictionary_gradient.Matrix();
  const auto& local_output_grad = get_local_prev_error_signals();
  const auto& local_width = local_input.Width();
  const auto& mini_batch_size = this->m_model->get_effective_mini_batch_size();

  // Update appropriate columns of gradient w.r.t. dictionary
  El::Zero(local_dict_grad);
  CPUMat dict_grad_v, output_grad_v;
  for (El::Int col = 0; col < local_width; ++ col) {
    const El::Int ind = static_cast<El::Int>(local_input(0, col));
    El::View(dict_grad_v, local_dict_grad, El::ALL, El::IR(ind));
    El::LockedView(output_grad_v, local_output_grad, El::ALL, El::IR(col));
    El::Axpy(DataType{1}, output_grad_v, dict_grad_v);
  }
  opt.add_to_gradient(m_dictionary_gradient,
                      DataType{1} / mini_batch_size,
                      true);

}

} // namespace lbann
