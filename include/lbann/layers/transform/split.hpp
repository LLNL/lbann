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

#ifndef LBANN_LAYER_SPLIT_HPP_INCLUDED
#define LBANN_LAYER_SPLIT_HPP_INCLUDED

#include <vector>
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class split_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  split_distconv_adapter(Layer& layer): data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~split_distconv_adapter() = default;
  void setup_distributions(tensor_overlap_constraints &constraints) override;
  dc::Shape get_activations_local_shape(int index) const override;
  std::unique_ptr<TensorDevType> setup_activations_i(int index) const override;
  void bp_compute();
};
#endif // LBANN_HAS_DISTCONV

/** @brief Present input tensor to multiple outputs. */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class split_layer : public data_type_layer<TensorDataType> {
public:

  split_layer(lbann_comm *comm) : data_type_layer<TensorDataType>(comm) {
    this->m_expected_num_child_layers = -1; // No limit on children
  }

  split_layer* copy() const override { return new split_layer(*this); }
  std::string get_type() const override { return "split"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    for (int i = 0; i < this->get_num_children(); ++i) {
      this->set_output_dims(this->get_input_dims(), i);
    }
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    const auto& input = this->get_prev_activations();
    for (int i = 0; i < this->get_num_children(); ++i) {
      El::LockedView(this->get_activations(i), input);
    }
  }

  void fp_compute() override {}

  void bp_compute() override {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      get_distconv_adapter().bp_compute();
      return;
    }
#endif // LBANN_HAS_DISTCONV
    auto& gradient_wrt_input = this->get_error_signals();
    if (this->get_num_children() > 0) {
      El::Copy(this->get_prev_error_signals(0), gradient_wrt_input);
    } else {
      El::Zero(gradient_wrt_input);
    }
    for (int i = 1; i < this->get_num_children(); ++i) {
      El::Axpy(DataType(1), this->get_prev_error_signals(i),
               gradient_wrt_input);
    }
  }

#ifdef LBANN_HAS_DISTCONV
 protected:
  bool is_distconv_supported() const override {
    return Dev == El::Device::GPU && T_layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override {
    this->get_distconv_adapter_ptr() = make_unique<split_distconv_adapter<
      TensorDataType, T_layout, Dev>>(*this);
  }
  split_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() override;
  const split_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
split_distconv_adapter<TensorDataType, T_layout, Dev>&
split_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() {
  return const_cast<split_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      static_cast<const split_layer<TensorDataType, T_layout, Dev>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const split_distconv_adapter<TensorDataType, T_layout, Dev>&
split_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const {
  return dynamic_cast<const split_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void split_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_distributions(tensor_overlap_constraints &constraints) {
  data_type_distconv_adapter<TensorDataType>::setup_distributions(
      constraints);

  auto &x = this->get_prev_activations_dist();
  auto &y = this->get_activations_dist();
  auto &dx = this->get_error_signals_dist();
  auto &dy = this->get_prev_error_signals_dist();

  constraints.mark_equivalent(x, y);
  constraints.mark_equivalent(dx, dy);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape split_distconv_adapter<TensorDataType, T_layout, Dev>::
get_activations_local_shape(int index) const {
  return data_type_distconv_adapter<TensorDataType>::get_activations_local_shape(0);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
std::unique_ptr<typename split_distconv_adapter<TensorDataType, T_layout, Dev>::TensorDevType>
split_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_activations_i(int index) const {
  return make_unique<TensorDevType>(this->get_prev_activations(0));
}
#endif // LBANN_HAS_DISTCONV

LBANN_DEFINE_LAYER_BUILDER(split);

#ifndef LBANN_SPLIT_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class split_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class split_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#ifdef LBANN_HAS_DISTCONV
#define PROTO_DEVICE(T, Device) \
  extern template class split_distconv_adapter<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class split_distconv_adapter<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_HAS_DISTCONV
#endif // LBANN_SPLIT_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_SPLIT_HPP_INCLUDED
