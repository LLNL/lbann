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

#define LBANN_RELU_LAYER_INSTANTIATE
#include "lbann/layers/activations/relu_impl.hpp"
#include "lbann/utils/entrywise_operator.hpp"

namespace lbann {

namespace {

/** Entry-wise operator. */
template <typename TensorDataType>
struct op
{
  inline TensorDataType operator()(const TensorDataType& x) const
  {
    return std::max(x, El::TypeTraits<TensorDataType>::Zero());
  }
};

/** Entry-wise operator for backprop.
 *  If the forward propagation step computes \f$ y = f(x) \f$, the
 *  backward propagation step computes
 *  \f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$.
 */
template <typename TensorDataType>
struct op_backprop
{
  inline TensorDataType operator()(const TensorDataType& x,
                                   const TensorDataType& dy) const
  {
    return x > El::TypeTraits<TensorDataType>::Zero()
             ? dy
             : El::TypeTraits<TensorDataType>::Zero();
  }
};

} // namespace

// Template instantiation
template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::fp_compute()
{
  apply_entrywise_unary_operator<op, TensorDataType>(
    this->get_prev_activations(),
    this->get_activations());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::bp_compute()
{
  apply_entrywise_binary_operator<op_backprop, TensorDataType>(
    this->get_activations(),
    this->get_prev_error_signals(),
    this->get_error_signals());
}

#define PROTO(T)                                                               \
  template class relu_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>;   \
  template class relu_layer<T, data_layout::MODEL_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
