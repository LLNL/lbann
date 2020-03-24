////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#define LBANN_IDENTITY_LAYER_INSTANTIATE
#include "lbann/layers/activations/identity.hpp"

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
void identity_distconv_adapter<TensorDataType, Layout, Device>::
setup_distributions(std::map<dc::Dist*, std::set<dc::Dist*>> &equivalents,
                    std::set<dc::Dist*> &updated,
                    std::set<dc::Dist*> &invariants) {
  data_type_distconv_adapter<TensorDataType>::setup_distributions(
      equivalents, updated, invariants);

  auto &x = this->get_prev_activations_dist();
  auto &y = this->get_activations_dist();
  auto &dx = this->get_error_signals_dist();
  auto &dy = this->get_prev_error_signals_dist();

  // x == y
  equivalents[&x].insert(&y);
  equivalents[&y].insert(&x);
  // dx == dy
  equivalents[&dx].insert(&dy);
  equivalents[&dy].insert(&dx);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void identity_distconv_adapter<TensorDataType, Layout, Device>::
setup_activations() {
  const auto &prev_activations = this->get_prev_activations();
  this->m_outputs.emplace_back(make_unique<TensorDevType>(
      prev_activations));
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void identity_distconv_adapter<TensorDataType, Layout, Device>::
setup_error_signals() {
  const auto &prev_error_signals = this->get_prev_error_signals();
  this->m_gradient_wrt_inputs.emplace_back(make_unique<TensorDevType>(
      prev_error_signals));
}
#endif // LBANN_HAS_DISTCONV

#define PROTO_DEVICE(T, Device) \
  template class identity_layer<T, data_layout::DATA_PARALLEL, Device>; \
  template class identity_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann
