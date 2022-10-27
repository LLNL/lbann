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

#ifndef LBANN_LAYER_BINARY_SWITCH_HPP_INCLUDED
#define LBANN_LAYER_BINARY_SWITCH_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Output tensor filled with a single value */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class binary_switch_layer : public data_type_layer<TensorDataType> {
public:

  binary_switch_layer(lbann_comm *comm,
                      std::vector<int> dims)
    : data_type_layer<TensorDataType>(comm) {
    this->set_output_dims(dims);
    this->m_expected_num_parent_layers = 0;
  }

  binary_switch_layer* copy() const override { return new binary_switch_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "binary_switch"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:

  friend class cereal::access;
  binary_switch_layer()
    : binary_switch_layer(nullptr, { 1 } )
  {}


  void fp_compute() override {
    if (this->is_frozen()) {
      El::Zero(this->get_activations());
    } else {
      El::Fill(this->get_activations(), El::To<TensorDataType>(1));
    }
  }

};


#ifndef LBANN_BINARY_SWITCH_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class binary_switch_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class binary_switch_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_BINARY_SWITCH_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_BINARY_SWITCH_HPP_INCLUDED
