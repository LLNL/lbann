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

#ifndef LBANN_LAYER_UNIFORM_HPP_INCLUDED
#define LBANN_LAYER_UNIFORM_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** Activations are drawn from uniform distribution.
 *  During validation and testing, the layer outputs the distribution
 *  mean.
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
    m_expected_num_parent_layers = 0;
  }
  uniform_layer* copy() const override { return new uniform_layer(*this); }
  std::string get_type() const override { return "uniform"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream ss;
    ss << "uniform_layer" << "  "
       << "min: " << m_min << " "
       << "max: " << m_max << " "
       << "dataLayout: " << this->get_data_layout_string(get_data_layout());
     return ss.str();
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
