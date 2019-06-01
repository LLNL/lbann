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

#ifndef LBANN_LAYER_CATEGORICAL_RANDOM_HPP_INCLUDED
#define LBANN_LAYER_CATEGORICAL_RANDOM_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** @brief Random categorical outputs.
 *
 *  Inputs are probability distributions and outputs are one-hot
 *  vectors. An input entry is the probability that the corresponding
 *  output entry is one.
 *
 *  @todo Remove.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class categorical_random_layer : public transform_layer {

 public:
  categorical_random_layer(lbann_comm *comm)
    : transform_layer(comm) {
    static_assert(Dev == El::Device::CPU,
                  "categorical random layer currently only supports CPU");
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "categorical random layer currently only supports DATA_PARALLEL");
  }
  categorical_random_layer* copy() const override { return new categorical_random_layer(*this); }
  std::string get_type() const override { return "categorical random"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:

  void fp_compute() override {

    // Input and output matrices
    const auto& input = get_prev_activations();
    const auto& local_input = input.LockedMatrix();
    auto& local_output = get_local_activations();
    const auto& width = input.Width();
    const auto& local_height = local_input.Height();
    const auto& local_width = local_input.Width();

    // Initialize output and random numbers
    const auto& mode = this->m_model->get_execution_mode();
    El::Zero(local_output);
    StarVCMat<El::Device::CPU> rand_mat(input.Grid(), input.Root());
    if (mode == execution_mode::training) {
      uniform_fill(rand_mat, 1, width, DataType(0.5), DataType(0.5));
    }

    // Process each mini-batch sample
    LBANN_OMP_PARALLEL_FOR
    for (El::Int col = 0; col < local_width; ++col) {

      // Determine index of output
      El::Int index = local_height - 1;
      if (mode == execution_mode::training) {
        // Choose first output with CDF above random number in (0,1)
        const auto& rand = rand_mat.GetLocal(0, col);
        DataType cdf = DataType(0);
        for (El::Int row = 0; row < local_height; ++row) {
          cdf += local_input(row, col);
          if (rand < cdf) {
            index = row;
            break;
          }
        }
      } else {
        // Choose mode of probability distribution
        const auto& input_ptr = local_input.LockedBuffer(0, col);
        index = (std::max_element(input_ptr, input_ptr + local_height)
                 - input_ptr);
      }

      // Output a one-hot vector
      local_output(index, col) = DataType(1);

    }

  }

};

} // namespace lbann

#endif // LBANN_LAYER_CATEGORICAL_RANDOM_HPP_INCLUDED
