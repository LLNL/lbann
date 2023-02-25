////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#define LBANN_MEAN_SQUARED_ERROR_LAYER_INSTANTIATE
#include "lbann/layers/loss/mean_squared_error.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
void local_fp_cpu(El::Int height,
                  const El::AbstractMatrix<TensorDataType>& local_prediction,
                  const El::AbstractMatrix<TensorDataType>& local_ground_truth,
                  El::AbstractMatrix<TensorDataType>& local_contribution) {

  // Useful constants
  const auto& local_height = local_prediction.Height();
  const auto& local_width = local_prediction.Width();

  // Compute local contribution to mean squared error
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    TensorDataType sum = El::TypeTraits<TensorDataType>::Zero();
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& err = (local_prediction(row, col)
                         - local_ground_truth(row, col));
      sum += err * err;
    }
    local_contribution(0, col) = sum / height;
  }

}

template <typename TensorDataType>
void local_bp_cpu(El::Int height,
                  const El::AbstractMatrix<TensorDataType>& local_prediction,
                  const El::AbstractMatrix<TensorDataType>& local_ground_truth,
                  const El::AbstractMatrix<TensorDataType>& local_gradient_wrt_output,
                  El::AbstractMatrix<TensorDataType>& local_gradient_wrt_prediction,
                  El::AbstractMatrix<TensorDataType>& local_gradient_wrt_ground_truth) {

  // Useful constants
  const TensorDataType scale = static_cast<TensorDataType>(TensorDataType(2) / height);
  const El::Int local_height = local_prediction.Height();
  const El::Int local_width = local_prediction.Width();

  // Compute gradients
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& err = (local_prediction(row, col)
                         - local_ground_truth(row, col));
      const auto& dy = local_gradient_wrt_output(0, col);
      local_gradient_wrt_prediction(row, col) = scale * err * dy;
      local_gradient_wrt_ground_truth(row, col) = - scale * err * dy;
    }
  }

}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void mean_squared_error_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  this->set_output_dims({1});

#ifdef LBANN_HAS_DISTCONV
  // In the current implementation of mean squared error in Distconv, we
  // do not use the reshape layer and just assumes both inputs have
  // the matching shape. Therefore, the following check on the input
  // dimensions would fail. We could address this by either 1)
  // implementing the reshape layer, or 2) giving a proper shape to
  // the ground-truth data.
  //
  if (this->distconv_enabled()) {
    return;
  }
#endif

  // Check that input dimensions match
  if (this->get_input_dims(0) != this->get_input_dims(1)) {
    const auto& parents = this->get_parent_layers();
    std::stringstream err;
    err << get_type() << " layer \"" << this->get_name() << "\" "
        << "has input tensors with different dimensions (";
    for (int i = 0; i < this->get_num_parents(); ++i) {
      const auto& dims = this->get_input_dims(i);
      err << (i > 0 ? ", " : "") << "layer \"" << parents[i]->get_name()
          << "\" outputs ";
      for (size_t j = 0; j < dims.size(); ++j) {
        err << (j > 0 ? " x " : "") << dims[j];
      }
    }
    err << ")";
    LBANN_ERROR(err.str());
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void mean_squared_error_layer<TensorDataType, Layout, Device>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Initialize workspace
  const auto& input_dist = this->get_prev_activations(0).DistData();
  m_workspace.reset(AbsDistMatrixType::Instantiate(
    *input_dist.grid,
    input_dist.root,
    El::STAR,
    input_dist.rowDist,
    (input_dist.blockHeight == 1 && input_dist.blockWidth == 1 ? El::ELEMENT
                                                               : El::BLOCK),
    input_dist.device));
#ifdef HYDROGEN_HAVE_CUB
  if (m_workspace->GetLocalDevice() == El::Device::GPU) {
    m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
  }
#endif // HYDROGEN_HAVE_CUB
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void mean_squared_error_layer<TensorDataType, Layout, Device>::fp_compute()
{

#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    fp_compute_distconv();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  // Initialize workspace
  m_workspace->Empty();
  m_workspace->AlignWith(this->get_prev_activations());
  m_workspace->Resize(1, this->get_prev_activations().Width());

  // Compute local contributions and accumulate
  /// @todo Consider reduce rather than allreduce
  local_fp_compute();
  this->get_comm()->allreduce(*m_workspace, m_workspace->RedundantComm());
  El::Copy(*m_workspace, this->get_activations());

  // Clean up
  m_workspace->Empty();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void mean_squared_error_layer<TensorDataType, Layout, Device>::bp_compute()
{

#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    bp_compute_distconv();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  // Initialize workspace
  m_workspace->Empty();
  m_workspace->AlignWith(this->get_prev_activations());
  El::Copy(this->get_prev_error_signals(), *m_workspace);

  // Compute local gradients
  local_bp_compute();

  // Clean up
  m_workspace->Empty();
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void mean_squared_error_layer<TensorDataType, T_layout, Dev>::local_fp_compute() {
  local_fp_cpu(this->get_input_size(),
               this->get_local_prev_activations(0),
               this->get_local_prev_activations(1),
               this->m_workspace->Matrix());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void mean_squared_error_layer<TensorDataType, T_layout, Dev>::local_bp_compute() {
  local_bp_cpu(this->get_input_size(),
               this->get_local_prev_activations(0),
               this->get_local_prev_activations(1),
               this->m_workspace->LockedMatrix(),
               this->get_local_error_signals(0),
               this->get_local_error_signals(1));
}

#define PROTO(T)                                      \
  template class mean_squared_error_layer<            \
    T, data_layout::DATA_PARALLEL, El::Device::CPU>;  \
  template class mean_squared_error_layer<            \
    T, data_layout::MODEL_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
