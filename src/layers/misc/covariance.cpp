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

#define LBANN_COVARIANCE_LAYER_INSTANTIATE
#include "lbann/layers/misc/covariance_impl.hpp"

namespace lbann {

namespace {

/** CPU forward prop implementation.
 *  We use a two-pass algorithm since it is more numerically stable
 *  than the naive single-pass algorithm.
 */
template <typename TensorDataType>
void fp_cpu(const El::AbstractDistMatrix<TensorDataType>& input0,
            const El::AbstractDistMatrix<TensorDataType>& input1,
            El::AbstractDistMatrix<TensorDataType>& output,
            El::AbstractDistMatrix<TensorDataType>& means,
            El::AbstractDistMatrix<TensorDataType>& workspace,
            bool biased)
{
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Local matrices
  const auto& local_input0 =
    static_cast<const CPUMatType&>(input0.LockedMatrix());
  const auto& local_input1 =
    static_cast<const CPUMatType&>(input1.LockedMatrix());
  auto& local_means = static_cast<CPUMatType&>(means.Matrix());
  auto& local_workspace = static_cast<CPUMatType&>(workspace.Matrix());

  // Dimensions
  const auto& height = input0.Height();
  const auto& width = input0.Width();
  const auto& local_height = local_input0.Height();
  const auto& local_width = local_input0.Width();

  // Compute column-wise mean
  means.Empty(false);
  means.AlignWith(input0);
  means.Resize(2, width);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    TensorDataType sum0 = El::TypeTraits<TensorDataType>::Zero(),
                   sum1 = El::TypeTraits<TensorDataType>::Zero();
    for (El::Int row = 0; row < local_height; ++row) {
      sum0 += local_input0(row, col);
      sum1 += local_input1(row, col);
    }
    local_means(0, col) = sum0 / height;
    local_means(1, col) = sum1 / height;
  }
  El::AllReduce(means, means.RedundantComm());

  // Compute column-wise covariance
  workspace.Empty(false);
  workspace.AlignWith(input0);
  workspace.Resize(1, width);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& mean0 = local_means(0, col);
    const auto& mean1 = local_means(1, col);
    TensorDataType sum = El::TypeTraits<TensorDataType>::Zero();
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& x0 = local_input0(row, col);
      const auto& x1 = local_input1(row, col);
      sum += (x0 - mean0) * (x1 - mean1);
    }
    local_workspace(0, col) = sum / (biased ? height : height - 1);
  }
  El::AllReduce(workspace, workspace.RedundantComm());
  El::Copy(workspace, output);
}

/** CPU backprop implementation.
 *  Means have already been computed in forward prop.
 */
template <typename TensorDataType>
void bp_cpu(const El::AbstractDistMatrix<TensorDataType>& input0,
            const El::AbstractDistMatrix<TensorDataType>& input1,
            const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
            El::AbstractDistMatrix<TensorDataType>& gradient_wrt_input0,
            El::AbstractDistMatrix<TensorDataType>& gradient_wrt_input1,
            const El::AbstractDistMatrix<TensorDataType>& means,
            El::AbstractDistMatrix<TensorDataType>& workspace,
            bool biased)
{
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Local matrices
  const auto& local_input0 =
    static_cast<const CPUMatType&>(input0.LockedMatrix());
  const auto& local_input1 =
    static_cast<const CPUMatType&>(input1.LockedMatrix());
  auto& local_gradient_wrt_input0 =
    static_cast<CPUMatType&>(gradient_wrt_input0.Matrix());
  auto& local_gradient_wrt_input1 =
    static_cast<CPUMatType&>(gradient_wrt_input1.Matrix());
  const auto& local_means =
    static_cast<const CPUMatType&>(means.LockedMatrix());
  auto& local_workspace = static_cast<CPUMatType&>(workspace.Matrix());

  // Dimensions
  const auto& height = input0.Height();
  const auto& local_height = local_input0.Height();
  const auto& local_width = local_input0.Width();

  // Initialize workspace with gradients w.r.t. output
  El::Copy(gradient_wrt_output, workspace);

  // Compute gradients w.r.t. input
  const TensorDataType scale =
    El::TypeTraits<TensorDataType>::One() /
    El::To<TensorDataType>(biased ? height : height - 1);
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& dy = local_workspace(0, col);
      const auto& x0 = local_input0(row, col);
      const auto& x1 = local_input1(row, col);
      const auto& mean0 = local_means(0, col);
      const auto& mean1 = local_means(1, col);
      auto& dx0 = local_gradient_wrt_input0(row, col);
      auto& dx1 = local_gradient_wrt_input1(row, col);
      dx0 = dy * scale * (x1 - mean1);
      dx1 = dy * scale * (x0 - mean0);
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void covariance_layer<TensorDataType, Layout, Device>::fp_compute()
{
  fp_cpu(this->get_prev_activations(0),
         this->get_prev_activations(1),
         this->get_activations(),
         *this->m_means,
         *this->m_workspace,
         this->m_biased);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void covariance_layer<TensorDataType, Layout, Device>::bp_compute()
{
  bp_cpu(this->get_prev_activations(0),
         this->get_prev_activations(1),
         this->get_prev_error_signals(),
         this->get_error_signals(0),
         this->get_error_signals(1),
         *this->m_means,
         *this->m_workspace,
         this->m_biased);
}

#define PROTO(T)                                                               \
  template class covariance_layer<T,                                           \
                                  data_layout::DATA_PARALLEL,                  \
                                  El::Device::CPU>;                            \
  template class covariance_layer<T,                                           \
                                  data_layout::MODEL_PARALLEL,                 \
                                  El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
