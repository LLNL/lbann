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
#ifndef LBANN_LAYERS_MISC_ROWWISE_WEIGHTS_NORMS_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_MISC_ROWWISE_WEIGHTS_NORMS_IMPL_HPP_INCLUDED

#include "lbann/layers/misc/rowwise_weights_norms.hpp"

namespace lbann {

template <typename T, data_layout L, El::Device D>
rowwise_weights_norms_layer<T, L, D>::rowwise_weights_norms_layer()
  : data_type_layer<T>(nullptr)
{
  this->m_expected_num_parent_layers = 0;
}

template <typename T, data_layout L, El::Device D>
auto rowwise_weights_norms_layer<T, L, D>::copy() const
  -> rowwise_weights_norms_layer*
{
  return new rowwise_weights_norms_layer(*this);
}

template <typename T, data_layout L, El::Device D>
std::string rowwise_weights_norms_layer<T, L, D>::get_type() const
{
  return "row-wise weights norms";
}

template <typename T, data_layout L, El::Device D>
data_layout rowwise_weights_norms_layer<T, L, D>::get_data_layout() const
{
  return L;
}

template <typename T, data_layout L, El::Device D>
El::Device rowwise_weights_norms_layer<T, L, D>::get_device_allocation() const
{
  return D;
}

template <typename T, data_layout L, El::Device D>
void rowwise_weights_norms_layer<T, L, D>::setup_dims()
{
  data_type_layer<T>::setup_dims();

  // Make sure weights have already been setup by another layer
  if (this->has_weights() != 1) {
    LBANN_ERROR("attempted to setup ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" with an invalid number of weights (expected 1, found ",
                this->num_weights(),
                ")");
  }
  if (this->get_weights(0).get_matrix_height() <= 0) {
    LBANN_ERROR("attempted to setup ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" with weights \"",
                this->get_weights(0).get_name(),
                "\" before weights have been setup (consider using hint_layer "
                "to force another layer to setup the weights first)");
  }

  // Output dimensions are height of weights matrix
  const auto& dims_ = this->get_weights(0).get_matrix_height_dims();
  std::vector<int> dims(dims_.begin(), dims_.end());
  this->set_output_dims(dims);
}

template <typename T, data_layout L, El::Device D>
void rowwise_weights_norms_layer<T, L, D>::fp_compute()
{

  // Weights data
  using WeightsType = data_type_weights<T>;
  const auto& w = dynamic_cast<const WeightsType&>(this->get_weights(0));
  const auto& weights_matrix = w.get_values();

  // Output tensor
  auto& output = this->get_activations();
  output.AlignWith(weights_matrix);
  if (weights_matrix.LocalHeight() != output.LocalHeight()) {
    LBANN_ERROR("data matrices for ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" and weights \"",
                w.get_name(),
                "\" are not aligned or have invalid layouts");
  }

  // Workspace buffers
  /// @todo Synchronize
  m_local_norms.Resize(weights_matrix.LocalHeight(), 1);
  LocalMat ones;
  El::Ones(ones, output.LocalWidth(), 1);

  // Compute norm of each row in weights matrix
  this->row_sqsums(weights_matrix.LockedMatrix(), m_local_norms);
  El::AllReduce(m_local_norms, weights_matrix.RowComm(), El::mpi::SUM);
  this->sqrt(m_local_norms);
  El::Gemm(El::NORMAL,
           El::TRANSPOSE,
           El::TypeTraits<T>::One(),
           m_local_norms,
           ones,
           El::TypeTraits<T>::Zero(),
           output.Matrix());
}

template <typename T, data_layout L, El::Device D>
void rowwise_weights_norms_layer<T, L, D>::bp_compute()
{

  // Weights data
  using WeightsType = data_type_weights<T>;
  auto& w = dynamic_cast<WeightsType&>(this->get_weights(0));
  const auto& weights_matrix = w.get_values();
  auto&& opt = w.get_optimizer();
  if (opt == nullptr) {
    return;
  }
  T alpha, beta;
  auto& weights_matrix_grad = opt->get_gradient_buffer(beta, alpha, false);

  // Gradient w.r.t. output tensor
  // Note: Assume output grad and weights data are aligned
  const auto& output_grad = this->get_prev_error_signals();
  if (weights_matrix.LocalHeight() != output_grad.LocalHeight()) {
    LBANN_ERROR("data matrices for ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" and weights \"",
                w.get_name(),
                "\" are not aligned or have invalid layouts");
  }

  // Workspace buffers
  LocalMat workspace(output_grad.LocalHeight(), 1);
  LocalMat ones;
  El::Ones(ones, output_grad.LocalWidth(), 1);

  // dw/dL = w / norm(w) * sum(dy/dL)
  El::Gemm(El::NORMAL,
           El::NORMAL,
           El::TypeTraits<T>::One(),
           output_grad.LockedMatrix(),
           ones,
           El::TypeTraits<T>::Zero(),
           workspace);
  El::AllReduce(workspace, output_grad.RowComm(), El::mpi::SUM);
  this->divide(workspace, m_local_norms);
  this->row_axpy(alpha,
                 workspace,
                 weights_matrix.LockedMatrix(),
                 beta,
                 weights_matrix_grad.Matrix());
}
} // namespace lbann
#endif // LBANN_LAYERS_MISC_ROWWISE_WEIGHTS_NORMS_IMPL_HPP_INCLUDED
