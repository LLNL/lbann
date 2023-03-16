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

#ifndef LBANN_LAYERS_LOSS_L2_NORM2_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_L2_NORM2_IMPL_HPP_INCLUDED

#include "lbann/comm.hpp"
#include "lbann/layers/loss/l2_norm2.hpp"

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void l2_norm2_layer<TensorDataType, T_layout, Dev>::fp_compute()
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

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void l2_norm2_layer<TensorDataType, T_layout, Dev>::bp_compute()
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

template <typename T, data_layout L, El::Device D>
void l2_norm2_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_l2_norm2();
}

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_L2_NORM2_IMPL_HPP_INCLUDED
