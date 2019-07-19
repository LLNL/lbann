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

#ifndef LBANN_LAYER_BERNOULLI_HPP_INCLUDED
#define LBANN_LAYER_BERNOULLI_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** @brief Random values with Bernoulli distribution.
 *
 *  During validation and testing, outputs are all zero.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class bernoulli_layer : public transform_layer {
private:
  /** Probability of outputting 1. */
  DataType m_prob;

public:
  bernoulli_layer(lbann_comm *comm,
                  std::vector<int> dims,
                  DataType prob = DataType(0.5))
    : transform_layer(comm), m_prob(prob) {
    set_output_dims(dims);
    this->m_expected_num_parent_layers = 0;
  }
  bernoulli_layer* copy() const override { return new bernoulli_layer(*this); }
  std::string get_type() const override { return "Bernoulli"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = transform_layer::get_description();
    desc.add("Probability", m_prob);
    return desc;
  }

protected:

  void fp_compute() override {
    auto& output = get_activations();
    if (this->m_model->get_execution_mode() == execution_mode::training) {
      bernoulli_fill(output, output.Height(), output.Width(), m_prob);
    } else {
      El::Zero(output);
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_BERNOULLI_HPP_INCLUDED
