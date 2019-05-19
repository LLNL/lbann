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

#ifndef LBANN_LAYER_DISCRETE_RANDOM_HPP_INCLUDED
#define LBANN_LAYER_DISCRETE_RANDOM_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** @brief Random output from discrete distribution.
 *
 *  Inputs are interpreted as the probability of choosing each
 *  distribution value.
 *
 *  @todo Remove.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class discrete_random_layer : public transform_layer {
 private:

  /** Values in discrete distribution. */
  std::vector<DataType> m_values;

 public:
  discrete_random_layer(lbann_comm *comm,
                        std::vector<DataType> values,
                        std::vector<int> dims)
    : transform_layer(comm),
      m_values(values) {
    static_assert(Dev == El::Device::CPU,
                  "discrete random layer currently only supports CPU");
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "discrete random layer currently only supports DATA_PARALLEL");
    set_output_dims(dims);
  }
  discrete_random_layer* copy() const override { return new discrete_random_layer(*this); }
  std::string get_type() const override { return "discrete random"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:

  void setup_dims() override {
    transform_layer::setup_dims();
    if (get_input_size() != (int) m_values.size()) {
      LBANN_ERROR("input tensor dimensions don't match number of "
                  "values in discrete distribution");
    }
  }

  void fp_compute() override {

    // Input and output matrices
    const auto& input = get_prev_activations();
    const auto& local_input = input.LockedMatrix();
    auto& output = get_activations();
    auto& local_output = output.Matrix();
    const int num_values = m_values.size();
    const auto& num_outputs = local_output.Height();
    const auto& width = input.Width();
    const auto& local_width = input.LocalWidth();

    // Initialize random numbers
    const auto& mode = this->m_model->get_execution_mode();
    if (mode == execution_mode::training) {
      uniform_fill(output, 1, width, DataType(0.5), DataType(0.5));
    }

    // Process each mini-batch sample
    LBANN_OMP_PARALLEL_FOR
    for (El::Int col = 0; col < local_width; ++col) {
      const auto& input_ptr = local_input.LockedBuffer(0, col);
      const auto& output_ptr = local_output.Buffer(0, col);
      if (mode == execution_mode::training) {
        // Sample outputs from probability distribution
        std::vector<DataType> cdf(num_values);
        std::partial_sum(input_ptr, input_ptr + num_values, cdf.begin());
        for (El::Int row = 0; row < num_outputs; ++row) {
          const int index = (std::lower_bound(cdf.begin(), cdf.end(),
                                              local_output(row, col))
                             - cdf.begin());
          local_output(row, col) = m_values[index];
        }
      } else {
        // Fill output with mode of probability distribution
        const int index = (std::max_element(input_ptr,
                                            input_ptr + num_values)
                           - input_ptr);
        std::fill_n(output_ptr, num_outputs, m_values[index]);
      }
    }

  }

};

} // namespace lbann

#endif // LBANN_LAYER_DISCRETE_RANDOM_HPP_INCLUDED
