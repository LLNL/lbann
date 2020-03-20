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

#include "lbann/layers/activations/relu.hpp"
#include "lbann/utils/cuda.hpp"

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

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::fp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    assert_always(Layout == data_layout::DATA_PARALLEL);
    fp_compute_distconv();
    return;
  }
#endif // LBANN_HAS_DISTCONV
  cuda::apply_entrywise_unary_operator<op, TensorDataType>(
      this->get_prev_activations(),
      this->get_activations());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::bp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    assert_always(Layout == data_layout::DATA_PARALLEL);
    bp_compute_distconv();
    return;
  }
#endif // LBANN_HAS_DISTCONV
  cuda::apply_entrywise_binary_operator<op_backprop, TensorDataType>(
      this->get_prev_activations(), this->get_prev_error_signals(),
      this->get_error_signals());
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::fp_compute_distconv() {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  assert_always(this->distconv_enabled());

  // Useful constants
  const TensorDataType one{1};
  const TensorDataType zero{0};

  dc().m_relu->forward(one, this->dc().get_prev_activations(),
                       zero, this->dc().get_activations());

  dc().copy_out_activations();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::bp_compute_distconv() {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  assert_always(this->distconv_enabled());

  const TensorDataType zero{0};
  const TensorDataType one{1};

  dc().m_relu->backward(one, this->dc().get_activations(),
                        this->dc().get_prev_error_signals(),
                        this->dc().get_prev_activations(),
                        zero, this->dc().get_error_signals());
  dc().copy_out_error_signals();
}
#endif // LBANN_HAS_DISTCONV

#define PROTO(T)                                                        \
  template class relu_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class relu_layer<T, data_layout::MODEL_PARALLEL, El::Device::GPU>;

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
