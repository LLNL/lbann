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

#ifndef LBANN_LAYERS_LOSS_MEAN_ABSOLUTE_ERROR_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_MEAN_ABSOLUTE_ERROR_IMPL_HPP_INCLUDED

#include "lbann/layers/loss/mean_absolute_error.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void mean_absolute_error_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();
  this->set_output_dims({1});

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
void mean_absolute_error_layer<TensorDataType, Layout, Device>::setup_data(
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
void mean_absolute_error_layer<TensorDataType, Layout, Device>::fp_compute()
{

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
void mean_absolute_error_layer<TensorDataType, Layout, Device>::bp_compute()
{

  // Initialize workspace
  m_workspace->Empty();
  m_workspace->AlignWith(this->get_prev_activations());
  El::Copy(this->get_prev_error_signals(), *m_workspace);

  // Compute local gradients
  local_bp_compute();

  // Clean up
  m_workspace->Empty();
}

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_MEAN_ABSOLUTE_ERROR_IMPL_HPP_INCLUDED
