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

#ifndef LBANN_LAYER_WEIGHTED_SUM_HPP_INCLUDED
#define LBANN_LAYER_WEIGHTED_SUM_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Add tensors with specified scaling factors. */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class weighted_sum_layer : public transform_layer {
private:

  /** Scaling factors for weighted sum. */
  std::vector<DataType> m_scaling_factors;

public:
  weighted_sum_layer(lbann_comm *comm,
                     std::vector<DataType> scaling_factors)
    : transform_layer(comm),
      m_scaling_factors(scaling_factors) {
    this->m_expected_num_parent_layers = -1; // No limit on parents
  }

  weighted_sum_layer* copy() const override { return new weighted_sum_layer(*this); }
  std::string get_type() const override { return "weighted sum"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = transform_layer::get_description();
    std::stringstream ss;
    for (size_t i = 0; i < m_scaling_factors.size(); ++i) {
      ss << (i > 0 ? ", " : "") << m_scaling_factors[i];
    }
    desc.add("Scaling factors", ss.str());
    return desc;
  }

protected:

  void setup_pointers() override {
    transform_layer::setup_pointers();
    std::stringstream err;
    if (get_num_parents() < 1) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has no parent layers";
      LBANN_ERROR(err.str());
    }
    if ((int) m_scaling_factors.size() != get_num_parents()) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has an invalid number of scaling factors "
          << "(found " << m_scaling_factors.size() << ", "
          << "but there are " << get_num_parents() << " parent layers)";
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
    El::Zero(output);
    for (int i = 0; i < get_num_parents(); ++i) {
      El::Axpy(m_scaling_factors[i], get_prev_activations(i), output);
    }
  }

  void bp_compute() override {
    const auto& gradient_wrt_output = get_prev_error_signals();
    for (int i = 0; i < get_num_parents(); ++i) {
      auto& gradient_wrt_input = get_error_signals(i);
      El::Zero(gradient_wrt_input);
      El::Axpy(m_scaling_factors[i], gradient_wrt_output,
               gradient_wrt_input);
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_WEIGHTED_SUM_HPP_INCLUDED
