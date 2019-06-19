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

#ifndef LBANN_LAYER_UNIFORM_HPP_INCLUDED
#define LBANN_LAYER_UNIFORM_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** @brief Random values with uniform distribution.
 *
 *  During validation and testing, outputs are all equal to the
 *  distribution mean.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class uniform_layer : public transform_layer {
private:
  /** Uniform distribution mean. */
  DataType m_min;
  /** Uniform distribution standard deviation. */
  DataType m_max;

public:

  uniform_layer(lbann_comm *comm,
                std::vector<int> dims,
                DataType min = DataType(0),
                DataType max = DataType(1))
    : transform_layer(comm), m_min(min), m_max(max) {
    set_output_dims(dims);
    this->m_expected_num_parent_layers = 0;
  }
  uniform_layer* copy() const override { return new uniform_layer(*this); }
  std::string get_type() const override { return "uniform"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto&& desc = transform_layer::get_description();
    std::stringstream ss;
    ss << "[" << m_min << "," << m_max << ")";
    desc.add("Range", ss.str());
    return desc;
  }

protected:

  void fp_compute() override {
    const auto& mean = (m_max + m_min) / 2;
    const auto& radius = (m_max - m_min) / 2;
    auto& output = get_activations();
    if (this->m_model->get_execution_mode() == execution_mode::training) {
      uniform_fill(output, output.Height(), output.Width(), mean, radius);
    } else {
      El::Fill(output, mean);
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_UNIFORM_HPP_INCLUDED
