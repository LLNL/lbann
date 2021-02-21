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

#ifndef LBANN_LAYER_SUM_HPP_INCLUDED
#define LBANN_LAYER_SUM_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class sum_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  sum_distconv_adapter(Layer& layer): data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~sum_distconv_adapter() = default;
  std::unique_ptr<TensorDevType> setup_error_signals_i(int index) const override;
  void fp_compute();
};
#endif // LBANN_HAS_DISTCONV

template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class sum_layer : public data_type_layer<TensorDataType> {
public:

  sum_layer(lbann_comm *comm)
    : data_type_layer<TensorDataType>(comm) {
    this->m_expected_num_parent_layers = -1; // No limit on parents
  }

  sum_layer* copy() const override { return new sum_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "sum"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:

  friend class cereal::access;
  sum_layer()
    : sum_layer(nullptr)
  {}

  void setup_pointers() override {
    data_type_layer<TensorDataType>::setup_pointers();
    if (this->get_num_parents() < 1) {
      std::stringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "has no parent layers";
      LBANN_ERROR(err.str());
    }
  }

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims(this->get_input_dims());

    // Check that input dimensions match
    const auto& output_dims = this->get_output_dims();
    for (int i = 0; i < this->get_num_parents(); ++i) {
      if (this->get_input_dims(i) != output_dims) {
        const auto& parents = this->get_parent_layers();
        std::stringstream err;
        err << get_type() << " layer \"" << this->get_name() << "\" "
            << "has input tensors with incompatible dimensions (";
        for (int j = 0; j < this->get_num_parents(); ++j) {
          const auto& dims = this->get_input_dims(j);
          err << (j > 0 ? ", " : "")
              << "layer \"" << parents[j]->get_name() << "\" outputs ";
          for (size_t k = 0; k < dims.size(); ++k) {
            err << (k > 0 ? " x " : "") << dims[k];
          }
        }
        err << ")";
        LBANN_ERROR(err.str());
      }
    }

  }

  void fp_compute() override {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      get_distconv_adapter().fp_compute();
      return;
    }
#endif // LBANN_HAS_DISTCONV
    auto& output = this->get_activations();
    El::Copy(this->get_prev_activations(0), output);
    for (int i = 1; i < this->get_num_parents(); ++i) {
      El::Axpy(DataType(1), this->get_prev_activations(i), output);
    }
  }

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override {
    const auto& gradient_wrt_output = this->get_prev_error_signals();
    for (int i = 0; i < this->get_num_parents(); ++i) {
      El::LockedView(this->get_error_signals(i), gradient_wrt_output);
    }
  }

  void bp_compute() override {}

#ifdef LBANN_HAS_DISTCONV
  friend class sum_distconv_adapter<TensorDataType, T_layout, Dev>;
 protected:
  bool is_distconv_supported() const override {
    return Dev == El::Device::GPU && T_layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override {
    this->get_distconv_adapter_ptr() = make_unique<sum_distconv_adapter<TensorDataType, T_layout, Dev>>(*this);
  }
  sum_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() override;
  const sum_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
sum_distconv_adapter<TensorDataType, T_layout, Dev>&
sum_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() {
  return const_cast<sum_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      static_cast<const sum_layer<TensorDataType, T_layout, Dev>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const sum_distconv_adapter<TensorDataType, T_layout, Dev>&
sum_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const {
  return dynamic_cast<const sum_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
std::unique_ptr<typename sum_distconv_adapter<TensorDataType, T_layout, Dev>::TensorDevType>
sum_distconv_adapter<TensorDataType, T_layout, Dev>::setup_error_signals_i(int index) const {
  return make_unique<TensorDevType>(this->get_prev_error_signals(0));
}
#endif // LBANN_HAS_DISTCONV

LBANN_DEFINE_LAYER_BUILDER(sum);

#ifndef LBANN_SUM_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class sum_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class sum_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#ifdef LBANN_HAS_DISTCONV
#define PROTO_DEVICE(T, Device) \
  extern template class sum_distconv_adapter<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class sum_distconv_adapter<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_HAS_DISTCONV
#endif // LBANN_SUM_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_SUM_HPP_INCLUDED
