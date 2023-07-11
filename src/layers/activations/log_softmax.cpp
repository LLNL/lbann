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

#define LBANN_LOG_SOFTMAX_LAYER_INSTANTIATE
#include "lbann/layers/activations/log_softmax.hpp"
#include "lbann/comm_impl.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
void fp(lbann_comm& comm,
        const El::AbstractDistMatrix<TensorDataType>& input,
        El::AbstractDistMatrix<TensorDataType>& output,
        El::AbstractDistMatrix<TensorDataType>& workspace)
{

  // Setup workspace
  workspace.Empty(false);
  workspace.AlignWith(input);
  workspace.Resize(1, input.Width());

  // Local matrices
  const auto& local_input =
    dynamic_cast<const CPUMatDT<TensorDataType>&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<CPUMatDT<TensorDataType>&>(output.Matrix());
  auto& local_workspace =
    dynamic_cast<CPUMatDT<TensorDataType>&>(workspace.Matrix());
  const auto local_height = local_input.Height();
  const auto local_width = local_input.Width();

  // Find column-wise maximum entries
  El::Fill(workspace, std::numeric_limits<TensorDataType>::lowest());
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    auto& max_entry = local_workspace(0, col);
    for (El::Int row = 0; row < local_height; ++row) {
      max_entry = std::max(max_entry, local_input(row, col));
    }
  }
  comm.allreduce(workspace, workspace.RedundantComm(), El::mpi::MAX);

  // Shift inputs and compute sum(exp(x)) for each column
  // Note: Shifting by the max prevents LogSumExp from blowing up.
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto shift = local_workspace(0, col);
    TensorDataType sum{0};
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& x = local_input(row, col);
      auto& y = local_output(row, col);
      y = x - shift;
      sum += std::exp(y);
    }
    local_workspace(0, col) = sum;
  }
  comm.allreduce(workspace, workspace.RedundantComm());

  // Compute output by subtracting LogSumExp
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const TensorDataType log_sum_exp = static_cast<TensorDataType>(
      std::log(local_workspace(El::TypeTraits<TensorDataType>::Zero(), col)));
    for (El::Int row = 0; row < local_height; ++row) {
      auto& y = local_output(row, col);
      y -= log_sum_exp;
    }
  }
}

template <typename TensorDataType>
void bp(lbann_comm& comm,
        const El::AbstractDistMatrix<TensorDataType>& output,
        const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
        El::AbstractDistMatrix<TensorDataType>& gradient_wrt_input,
        El::AbstractDistMatrix<TensorDataType>& workspace)
{

  // Local matrices
  const auto& local_output = dynamic_cast<const CPUMat&>(output.LockedMatrix());
  const auto& local_gradient_wrt_output =
    dynamic_cast<const CPUMat&>(gradient_wrt_output.LockedMatrix());
  auto& local_gradient_wrt_input =
    dynamic_cast<CPUMat&>(gradient_wrt_input.Matrix());
  auto& local_workspace = dynamic_cast<CPUMat&>(workspace.Matrix());
  const auto& local_height = local_output.Height();
  const auto& local_width = local_output.Width();

  // Compute sum of entries in gradient w.r.t. output
  El::Zero(workspace);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    auto& sum = local_workspace(0, col);
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& dy = local_gradient_wrt_output(row, col);
      sum += dy;
    }
  }
  comm.allreduce(workspace, workspace.RedundantComm());

  // Compute gradient w.r.t. input
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& sum = local_workspace(0, col);
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& y = local_output(row, col);
      const auto& dy = local_gradient_wrt_output(row, col);
      auto& dx = local_gradient_wrt_input(row, col);
      dx = dy - std::exp(y) * sum;
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void log_softmax_layer<TensorDataType, Layout, Device>::fp_compute()
{
  fp(*this->get_comm(),
     this->get_prev_activations(),
     this->get_activations(),
     *this->m_workspace);
}
template <typename TensorDataType, data_layout Layout, El::Device Device>
void log_softmax_layer<TensorDataType, Layout, Device>::bp_compute()
{
  bp(*this->get_comm(),
     this->get_activations(),
     this->get_prev_error_signals(),
     this->get_error_signals(),
     *this->m_workspace);
}

#define PROTO(T)                                                               \
  template class log_softmax_layer<T,                                          \
                                   data_layout::DATA_PARALLEL,                 \
                                   El::Device::CPU>;                           \
  template class log_softmax_layer<T,                                          \
                                   data_layout::MODEL_PARALLEL,                \
                                   El::Device::CPU>;

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
