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

#define LBANN_SELU_DROPOUT_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/selu_dropout.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/datatype_helpers.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

#include "lbann/proto/layers.pb.h"

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_selu_dropout_layer_from_pbuf(lbann_comm* comm,
                                          lbann_data::Layer const& proto_layer)
{
  auto const& params = proto_layer.selu_dropout();
  auto const keep_prob = El::To<T>(params.keep_prob());
  if (params.alpha() != 0.0 && params.scale() != 0.0) {
    auto const alpha = El::To<T>(params.alpha());
    auto const scale = El::To<T>(params.scale());
    return std::make_unique<selu_dropout<T, L, D>>(keep_prob, alpha, scale);
  }
  else {
    return std::make_unique<selu_dropout<T, L, D>>(keep_prob);
  }
}

namespace lbann {

template <typename T, data_layout L, El::Device D>
selu_dropout<T, L, D>::selu_dropout(T keep_prob, T alpha, T scale)
  : data_type_layer<T>(nullptr), m_keep_prob(keep_prob), m_mask(nullptr)
{
#ifdef LBANN_DETERMINISTIC
  LBANN_WARNING("selu_dropout: deterministic dropout not supported");
#endif
  // Compute alpha' and the affine transform.
  m_alpha_prime = -scale * alpha;
  m_a = keep_prob + m_alpha_prime * m_alpha_prime * keep_prob *
                      (El::TypeTraits<T>::One() - keep_prob);
  m_a = El::TypeTraits<T>::One() / El::Sqrt(m_a);
  m_b = -m_a * m_alpha_prime * (El::TypeTraits<T>::One() - keep_prob);
}

template <typename T, data_layout L, El::Device D>
selu_dropout<T, L, D>::selu_dropout(const selu_dropout& other)
  : data_type_layer<T>(other),
    m_alpha_prime(other.m_alpha_prime),
    m_a(other.m_a),
    m_b(other.m_b),
    m_keep_prob(other.m_keep_prob),
    m_mask(other.m_mask)
{
  if (m_mask != nullptr) {
    m_mask = m_mask->Copy();
  }
}

template <typename T, data_layout L, El::Device D>
auto selu_dropout<T, L, D>::operator=(const selu_dropout& other)
  -> selu_dropout&
{
  data_type_layer<T>::operator=(other);
  m_alpha_prime = other.m_alpha_prime;
  m_a = other.m_a;
  m_b = other.m_b;
  m_keep_prob = other.m_keep_prob;
  if (m_mask != nullptr) {
    delete m_mask;
  }
  m_mask = other.m_mask;
  if (m_mask != nullptr) {
    m_mask = m_mask->Copy();
  }
  return *this;
}

template <typename T, data_layout L, El::Device D>
selu_dropout<T, L, D>::~selu_dropout()
{
  if (m_mask != nullptr) {
    delete m_mask;
  }
}

template <typename T, data_layout L, El::Device D>
auto selu_dropout<T, L, D>::copy() const -> selu_dropout*
{
  return new selu_dropout(*this);
}

template <typename T, data_layout L, El::Device D>
std::string selu_dropout<T, L, D>::get_type() const
{
  return "selu dropout";
}

template <typename T, data_layout L, El::Device D>
data_layout selu_dropout<T, L, D>::get_data_layout() const
{
  return L;
}

template <typename T, data_layout L, El::Device D>
El::Device selu_dropout<T, L, D>::get_device_allocation() const
{
  return D;
}

template <typename T, data_layout L, El::Device D>
void selu_dropout<T, L, D>::write_specific_proto(lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_selu_dropout();
  msg->set_keep_prob(m_keep_prob);
  msg->set_alpha(-m_alpha_prime);
  msg->set_scale(El::To<T>(1));
}

template <typename T, data_layout L, El::Device D>
void selu_dropout<T, L, D>::setup_dims()
{
  data_type_layer<T>::setup_dims();
  this->set_output_dims(this->get_input_dims());
}

template <typename T, data_layout L, El::Device D>
void selu_dropout<T, L, D>::setup_data(size_t max_mini_batch_size)
{
  data_type_layer<T>::setup_data(max_mini_batch_size);
  if (m_mask != nullptr) {
    delete m_mask;
  }
  m_mask = this->get_activations().Copy();
}

template <typename T, data_layout L, El::Device D>
void selu_dropout<T, L, D>::fp_compute()
{
  if (this->m_model->get_execution_context().get_execution_mode() !=
        execution_mode::training ||
      m_keep_prob < El::To<T>(0.0f)) {
    // Do nothing if dropout is disabled
    El::Copy(this->get_prev_activations(), this->get_activations());
  }
  else {

    const auto* input_acts = &this->get_prev_activations();
    const El::Int height = input_acts->Height();
    const El::Int width = input_acts->Width();
    const El::Int local_height = input_acts->LocalHeight();
    const El::Int local_width = input_acts->LocalWidth();

    const auto& local_input_acts = input_acts->LockedMatrix();
    CPUMatrixType& local_output_acts = this->get_local_activations();
    CPUMatrixType& local_mask = m_mask->Matrix();

    // Construct and apply mask and the affine transform.
    // TODO: Optimize.
    El::Bernoulli(*m_mask, height, width, m_keep_prob);
    for (El::Int col = 0; col < local_width; ++col) {
      for (El::Int row = 0; row < local_height; ++row) {
        local_output_acts(row, col) =
          m_a * (local_input_acts(row, col) * local_mask(row, col) +
                 m_alpha_prime *
                   (El::TypeTraits<T>::One() - local_mask(row, col))) +
          m_b;
      }
    }
  }
}

template <typename T, data_layout L, El::Device D>
void selu_dropout<T, L, D>::bp_compute()
{
  if (this->m_model->get_execution_context().get_execution_mode() !=
        execution_mode::training ||
      m_keep_prob < El::To<T>(0.0f)) {
    El::Copy(this->get_prev_error_signals(), this->get_error_signals());
  }
  else {

    const auto& local_prev_error_signal = this->get_local_prev_error_signals();
    CPUMatrixType& local_error_signal = this->get_local_error_signals();
    CPUMatrixType& local_mask = m_mask->Matrix();
    const El::Int local_height = local_prev_error_signal.Height();
    const El::Int local_width = local_prev_error_signal.Width();
    // Reweight with the affine scale factor and the dropout mask.
    for (El::Int col = 0; col < local_width; ++col) {
      for (El::Int row = 0; row < local_height; ++row) {
        local_error_signal(row, col) =
          m_a * local_prev_error_signal(row, col) * local_mask(row, col);
      }
    }
  }
}

#define PROTO_DEVICE(T, Device)                                                \
  template class selu_dropout<T, data_layout::DATA_PARALLEL, Device>;          \
  template class selu_dropout<T, data_layout::MODEL_PARALLEL, Device>;         \
  LBANN_LAYER_BUILDER_ETI(selu_dropout, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
