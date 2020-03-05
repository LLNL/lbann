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
    if (!this->early_terminate_last_iteration()) {
      return;
    }
    // fall through the normal code path to obtain reference results
  }
#endif // LBANN_HAS_DISTCONV
  cuda::apply_entrywise_unary_operator<op, TensorDataType>(
      this->get_prev_activations(),
      this->get_activations());
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled() && this->early_terminate_last_iteration() &&
      this->keep_original()) {
    this->dump_reference_activations();
  }
#endif // LBANN_HAS_DISTCONV
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::bp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    assert_always(Layout == data_layout::DATA_PARALLEL);
    bp_compute_distconv();
    if (!this->early_terminate_last_iteration()) {
      return;
    }
  }
#endif // LBANN_HAS_DISTCONV
  cuda::apply_entrywise_binary_operator<op_backprop, TensorDataType>(
      this->get_prev_activations(), this->get_prev_error_signals(),
      this->get_error_signals());
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled() && this->early_terminate_last_iteration() &&
      this->keep_original()) {
    this->dump_reference_error_signals();
  }
#endif // LBANN_HAS_DISTCONV
}

#ifdef LBANN_HAS_DISTCONV
using dc::Dist;

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::init_distribution(
    std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
    std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
    std::set<dc::Dist*> &updated,
    std::set<dc::Dist*> &fixed)  {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  data_type_layer<TensorDataType>::init_distribution(
      dists, invariants, updated, fixed);
  if (!this->distconv_enabled()) return;
  auto &layer_dists = dists[this];
  // x == dx
  invariants[&layer_dists[0]].insert(
      &layer_dists[2]);
  invariants[&layer_dists[2]].insert(
      &layer_dists[0]);
  //y == dy
  invariants[&layer_dists[1]].insert(
      &layer_dists[3]);
  invariants[&layer_dists[3]].insert(
      &layer_dists[1]);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::
setup_tensors_fwd(const std::array<Dist, dc::num_dists> &dists) {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  data_type_layer<TensorDataType>::setup_tensors_fwd(dists);
  if (!this->distconv_enabled()) return;
  this->setup_prev_activations_tensor(dists);
  this->setup_activations_tensor(dists);
  this->setup_activations_copyout_tensor(dists);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::
setup_tensors_bwd(const std::array<Dist, dc::num_dists> &dists)  {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  data_type_layer<TensorDataType>::setup_tensors_bwd(dists);
  if (!this->distconv_enabled()) return;
  this->setup_prev_error_signals_tensor(dists);
  this->setup_error_signals_tensor(dists);
  this->setup_error_signals_copyout_tensor(dists);
  // Init the dc::Pooling layer
  m_relu = new dc::ReLU(dc::get_backend());
  m_relu->setup(this->get_prev_activations_t(),
                this->get_activations_t(),
                this->get_error_signals_t(),
                this->get_prev_error_signals_t());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::fp_compute_distconv() {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  assert_always(this->distconv_enabled());

  // Useful constants
  const TensorDataType one{1};
  const TensorDataType zero{0};

  m_relu->forward(one, this->get_prev_activations_t(),
                  zero, this->get_activations_t());

  this->copy_out_activations();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void relu_layer<TensorDataType, Layout, Device>::bp_compute_distconv() {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  assert_always(this->distconv_enabled());

  const TensorDataType zero{0};
  const TensorDataType one{1};

  m_relu->backward(one, this->get_activations_t(),
                   this->get_prev_error_signals_t(),
                   this->get_prev_activations_t(),
                   zero, this->get_error_signals_t());
  this->copy_out_error_signals();
}
#endif // LBANN_HAS_DISTCONV

template class relu_layer<
  DataType, data_layout::DATA_PARALLEL, El::Device::GPU>;
template class relu_layer<
  DataType, data_layout::MODEL_PARALLEL, El::Device::GPU>;

} // namespace lbann
