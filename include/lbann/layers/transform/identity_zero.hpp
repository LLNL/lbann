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

#ifndef LBANN_LAYER_IDENTITY_ZERO_HPP_INCLUDED
#define LBANN_LAYER_IDENTITY_ZERO_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Output tensor filled with a single value */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class identity_zero_layer : public data_type_layer<TensorDataType>
{
public:
  identity_zero_layer(lbann_comm* comm) : data_type_layer<TensorDataType>(comm)
  {
    this->m_expected_num_parent_layers = 1;
  }

  identity_zero_layer* copy() const override
  {
    return new identity_zero_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "identity_zero"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return true; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

protected:
  friend class cereal::access;
  identity_zero_layer() : identity_zero_layer(nullptr) {}

  void setup_dims() override;

  void fp_compute() override;

  void bp_compute() override;

  void write_specific_proto(lbann_data::Layer& proto) const final;
};

#ifndef LBANN_IDENTITY_ZERO_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class identity_zero_layer<T,                                 \
                                            data_layout::DATA_PARALLEL,        \
                                            Device>;                           \
  extern template class identity_zero_layer<T,                                 \
                                            data_layout::MODEL_PARALLEL,       \
                                            Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_IDENTITY_ZERO_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_IDENTITY_ZERO_HPP_INCLUDED
