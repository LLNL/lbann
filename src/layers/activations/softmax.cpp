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

#define LBANN_SOFTMAX_LAYER_INSTANTIATE
#include "lbann/comm_impl.hpp"
#include "lbann/layers/activations/softmax_impl.hpp"

#include "lbann/utils/dnn_lib/softmax.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
void fp_model_parallel(lbann_comm& comm,
                       const El::AbstractDistMatrix<TensorDataType>& input,
                       El::AbstractDistMatrix<TensorDataType>& output,
                       El::AbstractDistMatrix<TensorDataType>& workspace,
                       TensorDataType threshold_val,
                       softmax_mode mode)
{

  if (mode != softmax_mode::INSTANCE) {
    LBANN_ERROR("Unsupported softmax mode");
  }

  // Setup workspace
  workspace.Empty(false);
  workspace.AlignWith(input);
  workspace.Resize(1, input.Width());

  // Local matrices
  const auto& local_input = input.LockedMatrix();
  auto& local_output = output.Matrix();
  auto& local_workspace = workspace.Matrix();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

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

  // Exponentiate outputs and compute column sums
  // Note: Subtracting by the column max prevents output from blowing
  // up. Large negative values underflow to 0.
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto shift = local_workspace(0, col);
    TensorDataType sum = El::TypeTraits<TensorDataType>::Zero();
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& x = local_input(row, col);
      auto& y = local_output(row, col);
      y = std::exp(x - shift);
      sum += y;
    }
    local_workspace(0, col) = sum;
  }
  comm.allreduce(workspace, workspace.RedundantComm());

  // Divide outputs by column sums
  // Note: Small values can be rounded to minimum output value to
  // avoid denormalized floats.
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& scale = 1 / local_workspace(0, col);
    for (El::Int row = 0; row < local_height; ++row) {
      auto& y = local_output(row, col);
      y = scale * y;
#ifdef LBANN_ENABLE_SOFTMAX_THRESHOLD
      y = std::max(y, threshold_val);
#endif // LBANN_ENABLE_SOFTMAX_THRESHOLD
    }
  }
}

template <typename TensorDataType>
void bp_model_parallel(
  lbann_comm& comm,
  const El::AbstractDistMatrix<TensorDataType>& output,
  const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
  El::AbstractDistMatrix<TensorDataType>& gradient_wrt_input,
  El::AbstractDistMatrix<TensorDataType>& workspace,
  TensorDataType threshold_val,
  softmax_mode mode)
{

  if (mode != softmax_mode::INSTANCE) {
    LBANN_ERROR("Unsupported softmax mode");
  }

  // Local matrices
  const auto& local_output = output.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();
  auto& local_workspace = workspace.Matrix();
  const auto& local_height = local_output.Height();
  const auto& local_width = local_output.Width();

  // Compute dot products between output and gradient w.r.t. output
  El::Zero(local_workspace);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    auto& y_dot_dy = local_workspace(0, col);
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& y = local_output(row, col);
      const auto& dy = local_gradient_wrt_output(row, col);
      y_dot_dy += y * dy;
    }
  }
  comm.allreduce(workspace, workspace.RedundantComm());

  // Compute gradient w.r.t. input
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& y_dot_dy = local_workspace(0, col);
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& y = local_output(row, col);
      const auto& dy = local_gradient_wrt_output(row, col);
      auto& dx = local_gradient_wrt_input(row, col);
      dx = y * (dy - y_dot_dy);
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void softmax_layer<TensorDataType, Layout, Device>::fp_compute()
{
  if constexpr (Layout == data_layout::DATA_PARALLEL) {
    const auto& local_input =
      dynamic_cast<El::Matrix<TensorDataType, El::Device::CPU> const&>(
        this->get_prev_activations().LockedMatrix());
    auto& local_output =
      dynamic_cast<El::Matrix<TensorDataType, El::Device::CPU>&>(
        this->get_activations().Matrix());
    this->input_descriptor_.set(
      dnn_backend::template data_type<TensorDataType>(),
      std::vector<El::Int>{local_input.Width(), local_input.Height()},
      std::vector<El::Int>{local_input.LDim(), El::To<El::Int>(1)});
    this->output_descriptor_.set(
      dnn_backend::template data_type<TensorDataType>(),
      std::vector<El::Int>{local_output.Width(), local_output.Height()},
      std::vector<El::Int>{local_output.LDim(), El::To<El::Int>(1)});
    dnn_lib::softmax_forward(1.f,
                             this->input_descriptor_,
                             local_input,
                             0.f,
                             this->output_descriptor_,
                             local_output,
                             this->m_mode);
  }
  else
    fp_model_parallel(*this->get_comm(),
                      this->get_prev_activations(),
                      this->get_activations(),
                      *this->m_workspace,
                      this->threshold_val,
                      this->m_mode);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void softmax_layer<TensorDataType, Layout, Device>::bp_compute()
{
  if constexpr (Layout == data_layout::DATA_PARALLEL) {
    auto const& local_output =
      dynamic_cast<El::Matrix<TensorDataType, El::Device::CPU> const&>(
        this->get_activations().LockedMatrix());
    auto const& local_grad_wrt_output =
      dynamic_cast<El::Matrix<TensorDataType, El::Device::CPU> const&>(
        this->get_prev_error_signals().LockedMatrix());
    auto& local_grad_wrt_input =
      dynamic_cast<El::Matrix<TensorDataType, El::Device::CPU>&>(
        this->get_error_signals().Matrix());
    auto const data_type = dnn_backend::template data_type<TensorDataType>();
    grad_wrt_output_descriptor_.set(
      data_type,
      std::vector<El::Int>{local_grad_wrt_output.Width(),
                           local_grad_wrt_output.Height()},
      std::vector<El::Int>{local_grad_wrt_output.LDim(), El::To<El::Int>(1)});
    grad_wrt_input_descriptor_.set(
      data_type,
      std::vector<El::Int>{local_grad_wrt_input.Width(),
                           local_grad_wrt_input.Height()},
      std::vector<El::Int>{local_grad_wrt_input.LDim(), El::To<El::Int>(1)});
    dnn_lib::softmax_backward(1.f,
                              this->output_descriptor_,
                              local_output,
                              this->grad_wrt_output_descriptor_,
                              local_grad_wrt_output,
                              0.f,
                              this->grad_wrt_input_descriptor_,
                              local_grad_wrt_input,
                              this->m_mode);
  }
  else
    bp_model_parallel(*this->get_comm(),
                      this->get_activations(),
                      this->get_prev_error_signals(),
                      this->get_error_signals(),
                      *this->m_workspace,
                      this->threshold_val,
                      this->m_mode);
}

#define PROTO(T)                                                               \
  template class softmax_layer<T,                                              \
                               data_layout::DATA_PARALLEL,                     \
                               El::Device::CPU>;                               \
  template class softmax_layer<T, data_layout::MODEL_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
