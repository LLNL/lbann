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

#ifndef LBANN_LAYER_CONSTANT_HPP_INCLUDED
#define LBANN_LAYER_CONSTANT_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** @brief Constant output. */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class constant_layer : public transform_layer {
public:

  constant_layer(lbann_comm *comm,
                 DataType value,
                 std::vector<int> dims)
    : transform_layer(comm), m_value(value) {
    set_output_dims(dims);
    this->m_expected_num_parent_layers = 0;
  }

  constant_layer* copy() const override { return new constant_layer(*this); }
  std::string get_type() const override { return "constant"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = transform_layer::get_description();
    desc.add("Value", m_value);
    return desc;
  }

protected:

  void fp_compute() override {
    if (m_value == EvalType(0)) {
      El::Zero(get_activations());
    } else {
      El::Fill(get_activations(), m_value);
    }
  }

private:

  /** Constant value. */
  DataType m_value;

};

} // namespace lbann

#endif // LBANN_LAYER_CONSTANT_HPP_INCLUDED
