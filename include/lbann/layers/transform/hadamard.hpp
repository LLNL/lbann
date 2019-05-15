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

#ifndef LBANN_LAYER_HADAMARD_HPP_INCLUDED
#define LBANN_LAYER_HADAMARD_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Entry-wise tensor product. */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class hadamard_layer : public transform_layer {
public:

  hadamard_layer(lbann_comm *comm) : transform_layer(comm) {
    this->m_expected_num_parent_layers = -1; // No limit on parents
  }

  hadamard_layer* copy() const override { return new hadamard_layer(*this); }
  std::string get_type() const override { return "Hadamard"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:

  void setup_pointers() override {
    transform_layer::setup_pointers();
    if (get_num_parents() < 1) {
      std::stringstream err;
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has no parent layers";
      LBANN_ERROR(err.str());
    }
  }

  void setup_dims() override {
    transform_layer::setup_dims();
    set_output_dims(get_input_dims());

    // Check that input dimensions match
    const auto& output_dims = get_output_dims();
    for (int i = 0; i < get_num_parents(); ++i) {
      if (get_input_dims(i) != output_dims) {
        const auto& parents = get_parent_layers();
        std::stringstream err;
        err << get_type() << " layer \"" << get_name() << "\" "
            << "has input tensors with incompatible dimensions (";
        for (int j = 0; j < get_num_parents(); ++j) {
          const auto& dims = get_input_dims(j);
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
    auto& output = get_activations();
    switch (get_num_parents()) {
    case 0: El::Fill(output, DataType(1)); break;
    case 1: El::LockedView(output, get_prev_activations()); break;
    default:
      El::Hadamard(get_prev_activations(0),
                   get_prev_activations(1),
                   output);
      for (int i = 2; i < get_num_parents(); ++i) {
        El::Hadamard(get_prev_activations(i), output, output);
      }
    }
  }

  void bp_compute() override {
    const int num_parents = get_num_parents();
    const auto& gradient_wrt_output = get_prev_error_signals();
    switch (num_parents) {
    case 0: break;
    case 1:
      El::LockedView(get_error_signals(), gradient_wrt_output);
      break;
    default:
      for (int i = 0; i < num_parents; ++i) {
        auto& gradient_wrt_input = get_error_signals(i);
        El::Copy(gradient_wrt_output, gradient_wrt_input);
        for (int j = 0; j < num_parents; ++j) {
          if (i != j) {
            El::Hadamard(get_prev_activations(j),
                         gradient_wrt_input,
                         gradient_wrt_input);
          }
        }
      }
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_HADAMARD_HPP_INCLUDED
