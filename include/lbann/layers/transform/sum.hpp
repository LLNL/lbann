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

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#endif

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType>
class sum_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  sum_distconv_adapter(Layer& layer): data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~sum_distconv_adapter() = default;
  void setup_error_signals() override;
};
#endif // LBANN_HAS_DISTCONV


template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class sum_layer : public transform_layer<TensorDataType> {
public:

  sum_layer(lbann_comm *comm)
    : transform_layer<TensorDataType>(comm) {
    this->m_expected_num_parent_layers = -1; // No limit on parents
  }

  sum_layer* copy() const override { return new sum_layer(*this); }
  std::string get_type() const override { return "sum"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:

  void setup_pointers() override {
    transform_layer<TensorDataType>::setup_pointers();
    if (this->get_num_parents() < 1) {
      std::stringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "has no parent layers";
      LBANN_ERROR(err.str());
    }
  }

  void setup_dims() override {
    transform_layer<TensorDataType>::setup_dims();
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

  void fp_compute() override;

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override {
    const auto& gradient_wrt_output = this->get_prev_error_signals();
    for (int i = 0; i < this->get_num_parents(); ++i) {
      El::LockedView(this->get_error_signals(i), gradient_wrt_output);
    }
  }

  void bp_compute() override {}

#ifdef LBANN_HAS_DISTCONV
  friend class sum_distconv_adapter<TensorDataType>;
 protected:
  void setup_distconv_adapter() override {
    this->get_dc() = make_unique<sum_distconv_adapter<TensorDataType>>(*this);
  }
#endif // LBANN_HAS_DISTCONV

};

template <typename TensorDataType>
void sum_distconv_adapter<TensorDataType>::setup_error_signals() {
  this->m_gradient_wrt_inputs.clear();
  this->m_gradient_wrt_inputs.resize(this->layer().get_num_parents());
  for (int i = 0; i < this->layer().get_num_parents(); ++i) {
    this->m_gradient_wrt_inputs[i]= make_unique<TensorDevType>(
        this->get_prev_error_signals(0));
  }
}

LBANN_DEFINE_LAYER_BUILDER(sum);

#ifndef LBANN_SUM_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class sum_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class sum_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SUM_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_SUM_HPP_INCLUDED
