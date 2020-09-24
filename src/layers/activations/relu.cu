////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
#include "lbann/layers/activations/relu.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

/** Entry-wise operator. */
template <typename TensorDataType>
struct op {
  inline __device__ TensorDataType operator()(TensorDataType x) const {
    return x > TensorDataType{0} ? x : TensorDataType{0};
  }
};

/** Entry-wise operator for backprop.
 *  If the forward propagation step computes \f$ y = f(x) \f$, the
 *  backward propagation step computes
 *  \f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$.
 */
template <typename TensorDataType>
struct op_backprop {
  inline __device__ TensorDataType operator()(TensorDataType x, TensorDataType dy) const {
    return x > TensorDataType{0} ? dy : TensorDataType{0};
  }
};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
void fp_compute_distconv(relu_distconv_adapter<TensorDataType, Layout, Device> &dc) {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  dc.m_relu->forward(TensorDataType{1}, dc.get_prev_activations(),
                     TensorDataType{0}, dc.get_activations());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void bp_compute_distconv(relu_distconv_adapter<TensorDataType, Layout, Device> &dc) {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  dc.m_relu->backward(TensorDataType{1}, dc.get_activations(),
                      dc.get_prev_error_signals(),
                      dc.get_prev_activations(),
                      TensorDataType{0}, dc.get_error_signals());
}
#endif // LBANN_HAS_DISTCONV
} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::fp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    fp_compute_distconv(get_distconv_adapter());
    return;
  }
#endif // LBANN_HAS_DISTCONV
  gpu_lib::apply_entrywise_unary_operator<op, TensorDataType>(
      this->get_prev_activations(),
      this->get_activations());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::bp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    bp_compute_distconv(get_distconv_adapter());
    return;
  }
#endif // LBANN_HAS_DISTCONV
  gpu_lib::apply_entrywise_binary_operator<op_backprop, TensorDataType>(
      this->get_prev_activations(), this->get_prev_error_signals(),
      this->get_error_signals());
}

#define PROTO(T)                                                        \
  template class relu_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class relu_layer<T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
