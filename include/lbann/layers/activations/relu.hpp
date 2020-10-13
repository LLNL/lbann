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

#ifndef LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class relu_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  relu_distconv_adapter(Layer& layer): data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~relu_distconv_adapter() = default;
  void setup_distributions(tensor_overlap_constraints &constraints) override;
  void setup_layer(size_t workspace_capacity) override;
  std::unique_ptr<dc::ReLU> m_relu;
};
#endif // LBANN_HAS_DISTCONV

/** Rectified linear unit activation function layer.
 *  \f[ ReLU(x) = \text{max}(x, 0) \f]
 *  See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class relu_layer : public data_type_layer<TensorDataType> {
public:
  relu_layer(lbann_comm *comm) : data_type_layer<TensorDataType>(comm) {}
  relu_layer* copy() const override { return new relu_layer(*this); }
  std::string get_type() const override { return "ReLU"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:
  void fp_compute() override;
  void bp_compute() override;
#ifdef LBANN_HAS_DISTCONV
  bool is_distconv_supported() const override {
    return Dev == El::Device::GPU && T_layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override {
    this->get_distconv_adapter_ptr() = make_unique<relu_distconv_adapter<
      TensorDataType, T_layout, Dev>>(*this);
  }
  relu_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() override;
  const relu_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
relu_distconv_adapter<TensorDataType, T_layout, Dev>&
relu_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() {
  return const_cast<relu_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      static_cast<const relu_layer<TensorDataType, T_layout, Dev>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const relu_distconv_adapter<TensorDataType, T_layout, Dev>&
relu_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const {
  return dynamic_cast<const relu_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void relu_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_distributions(tensor_overlap_constraints &constraints) {
  data_type_distconv_adapter<TensorDataType>::setup_distributions(
      constraints);

  auto &x = this->get_prev_activations_dist();
  auto &y = this->get_activations_dist();
  auto &dx = this->get_error_signals_dist();
  auto &dy = this->get_prev_error_signals_dist();

  // x == dx
  constraints.mark_equivalent(x, dx);
  // y == dy
  constraints.mark_equivalent(y, dy);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void relu_distconv_adapter<TensorDataType, T_layout, Dev>::setup_layer(
    size_t workspace_capacity) {
  m_relu = make_unique<dc::ReLU>(dc::get_backend());
  m_relu->setup(this->get_prev_activations(),
                this->get_activations(),
                this->get_error_signals(),
                this->get_prev_error_signals());
}
#endif // LBANN_HAS_DISTCONV

#ifndef LBANN_RELU_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class relu_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class relu_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_RELU_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
