////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#define LBANN_DROPOUT_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/dropout.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

template <typename TensorDataType, data_layout layout, El::Device device>
void dropout<TensorDataType, layout, device>::fp_compute_cpu() {

  // Matrices
  const auto& input = this->get_prev_activations();
  auto& output = this->get_activations();

  // Do nothing if dropout is disabled
  const auto& mode =
    this->m_model->get_execution_context().get_execution_mode();
  if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
    El::Copy(input, output);
    return;
  }

  // Construct mask matrix
  const TensorDataType scale = static_cast<TensorDataType>(1 / m_keep_prob);
  const auto& height = input.Height();
  const auto& width = input.Width();
  m_mask->Resize(height, width);
#ifdef LBANN_DETERMINISTIC
  bernoulli_fill_procdet(*m_mask, height, width, TensorDataType(m_keep_prob));
  El::Scale(scale, *m_mask);
#else
  El::EntrywiseMap(
    *m_mask,
    (std::function<TensorDataType(
       const TensorDataType&)>)([this, scale](
                                  const TensorDataType& z) -> TensorDataType {
      auto& gen = get_fast_generator();
      std::bernoulli_distribution dist(m_keep_prob);
      return dist(gen) ? scale : El::TypeTraits<TensorDataType>::Zero();
    }));
#endif // LBANN_DETERMINISTIC

  // Apply mask matrix to get activations
  El::Hadamard(input, *m_mask, output);
}

/** Adjust gradients for dropout in backprop. */
template <typename TensorDataType, data_layout layout, El::Device device>
void dropout<TensorDataType, layout, device>::bp_compute_cpu() {
  const auto& gradient_wrt_output = this->get_prev_error_signals();
  auto& gradient_wrt_input = this->get_error_signals();
  const auto& mode = this->m_model->get_execution_context().get_execution_mode();
  if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
    El::Copy(gradient_wrt_output, gradient_wrt_input);
  } else {
    El::Hadamard(gradient_wrt_output, *m_mask, gradient_wrt_input);
  }
}

template <typename TensorDataType, data_layout layout, El::Device device>
std::unique_ptr<Layer> build_dropout_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const& layer_msg)
{
  const auto& params = layer_msg.dropout();
  return std::make_unique<dropout_layer<TensorDataType, layout, device>>(
    params.keep_prob());
}

template <typename T, data_layout L, El::Device D>
void dropout<T,L,D>::write_specific_proto(lbann_data::Layer& proto) const {
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_dropout();
  msg->set_keep_prob(m_keep_prob);
}

#define PROTO(T)                                         \
  template std::unique_ptr<Layer>                                       \
  build_dropout_layer_from_pbuf<T, data_layout::DATA_PARALLEL, El::Device::CPU>( \
    lbann_comm*, lbann_data::Layer const&);                             \
  template std::unique_ptr<Layer>                                       \
  build_dropout_layer_from_pbuf<T, data_layout::MODEL_PARALLEL, El::Device::CPU>( \
    lbann_comm*, lbann_data::Layer const&);                             \
  template class dropout<T, data_layout::DATA_PARALLEL, El::Device::CPU>; \
  template class dropout<T, data_layout::MODEL_PARALLEL, El::Device::CPU>

#include "lbann/macros/instantiate.hpp"

}// namespace lbann
