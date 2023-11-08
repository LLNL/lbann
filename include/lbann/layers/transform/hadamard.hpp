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

#ifndef LBANN_LAYER_HADAMARD_HPP_INCLUDED
#define LBANN_LAYER_HADAMARD_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"
#include <vector>

namespace lbann {

/** @brief Entry-wise tensor product */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class hadamard_layer : public data_type_layer<TensorDataType>
{
public:
  hadamard_layer(lbann_comm* comm) : data_type_layer<TensorDataType>(comm)
  {
    this->m_expected_num_parent_layers = -1; // No limit on parents
  }

  hadamard_layer* copy() const override { return new hadamard_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "Hadamard"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return true; }
  int get_backprop_requirements() const override
  {
    if (this->get_num_parents() > 1)
      return ERROR_SIGNALS | PREV_ACTIVATIONS;
    return ERROR_SIGNALS;
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  hadamard_layer() : hadamard_layer(nullptr) {}

  void setup_pointers() override
  {
    data_type_layer<TensorDataType>::setup_pointers();
    if (this->get_num_parents() < 1) {
      std::stringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "has no parent layers";
      LBANN_ERROR(err.str());
    }
  }

  void setup_dims() override
  {
    data_type_layer<TensorDataType>::setup_dims();
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
          err << (j > 0 ? ", " : "") << "layer \"" << parents[j]->get_name()
              << "\" outputs ";
          for (size_t k = 0; k < dims.size(); ++k) {
            err << (k > 0 ? " x " : "") << dims[k];
          }
        }
        err << ")";
        LBANN_ERROR(err.str());
      }
    }
  }

  void fp_compute() override
  {
    auto& output = this->get_activations();
    switch (this->get_num_parents()) {
    case 0:
      El::Fill(output, El::TypeTraits<TensorDataType>::One());
      break;
    case 1:
      El::LockedView(output, this->get_prev_activations());
      break;
    default:
      El::Hadamard(this->get_prev_activations(0),
                   this->get_prev_activations(1),
                   output);
      for (int i = 2; i < this->get_num_parents(); ++i) {
        El::Hadamard(this->get_prev_activations(i), output, output);
      }
    }
  }

  void bp_compute() override
  {
    const int num_parents = this->get_num_parents();
    const auto& gradient_wrt_output = this->get_prev_error_signals();
    switch (num_parents) {
    case 0:
      break;
    case 1:
      El::LockedView(this->get_error_signals(), gradient_wrt_output);
      break;
    default:
      for (int i = 0; i < num_parents; ++i) {
        auto& gradient_wrt_input = this->get_error_signals(i);
        El::Copy(gradient_wrt_output, gradient_wrt_input);
        for (int j = 0; j < num_parents; ++j) {
          if (i != j) {
            El::Hadamard(this->get_prev_activations(j),
                         gradient_wrt_input,
                         gradient_wrt_input);
          }
        }
      }
    }
  }
};

#ifndef LBANN_HADAMARD_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class hadamard_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class hadamard_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_HADAMARD_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_HADAMARD_HPP_INCLUDED
