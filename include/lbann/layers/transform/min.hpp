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

#ifndef LBANN_LAYER_MIN_HPP_INCLUDED
#define LBANN_LAYER_MIN_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** Min layer.
 *  This layer outputs the entrywise minimum of its input tensors.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class min_layer : public transform_layer {

 public:
  min_layer(lbann_comm *comm)
    : transform_layer(comm) {

    /// @todo Implement
    static_assert(Dev == El::Device::CPU,
                  "min layer currently only supports CPU");

    // Min layer has no limit on parents
    m_expected_num_parent_layers = -1;

  }

  min_layer* copy() const override { return new min_layer(*this); }
  std::string get_type() const override { return "min"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:

  void setup_pointers() override {
    transform_layer::setup_pointers();
    if (get_num_parents() < 1) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" has no parents, "
          << "but min layers expect at least one parent";
      LBANN_ERROR(err.str());
    }
  }

  void setup_dims() override {
    transform_layer::setup_dims();
    const auto& output_dims = get_output_dims();
    for (int i = 0; i < get_num_parents(); ++i) {
      const auto& input_dims = get_input_dims(i);
      if (input_dims != output_dims) {
        std::stringstream err;
        err << get_type() << " layer \"" << get_name() << "\" "
            << "expects input tensors with dimensions ";
        for (size_t j = 0; j < output_dims.size(); ++j) {
          err << (j > 0 ? " x " : "") << output_dims[j];
        }
        err << ", but parent layer "
            << "\"" << m_parent_layers[i]->get_name() << "\" "
            << "outputs with dimensions ";
        for (size_t j = 0; j < input_dims.size(); ++j) {
          err << (j > 0 ? " x " : "") << input_dims[j];
        }
        LBANN_ERROR(err.str());
      }
    }
  }

  void fp_compute() override {

    // Handle case with one parent
    // Note: Case with no parents is handled in setup_pointers
    const int num_parents = get_num_parents();
    if (num_parents == 1) {
      El::LockedView(get_activations(), get_prev_activations());
      return;
    }

    // Local matrices
    const auto& local_input0 = get_local_prev_activations(0);
    const auto& local_input1 = get_local_prev_activations(1);
    auto& local_output = get_local_activations();
    const int local_height = local_output.Height();
    const int local_width = local_output.Width();

    // Minimum of first two inputs
    LBANN_OMP_TASKLOOP_COLLAPSE2
    for (int col = 0; col < local_width; ++col) {
      for (int row = 0; row < local_height; ++row) {
        local_output(row, col) = std::min(local_input0(row, col),
                                          local_input1(row, col));
      }
    }

    // Handle case with more than two parents
    for (int i = 2; i < num_parents; ++i) {
      const auto& local_input = get_local_prev_activations(i);
      LBANN_OMP_TASKLOOP_COLLAPSE2
      for (int col = 0; col < local_width; ++col) {
        for (int row = 0; row < local_height; ++row) {
          const auto& x = local_input(row, col);
          auto& y = local_output(row, col);
          if (x < y) { y = x; }
        }
      }
    }

  }

  void bp_compute() override {

    // Useful constants
    const DataType zero = DataType(0);

    // Local matrices
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    const int local_height = local_gradient_wrt_output.Height();
    const int local_width = local_gradient_wrt_output.Width();

    // Handle cases for different number of parents
    // Note: Case with no parents is handled in setup_pointers
    const int num_parents = get_num_parents();
    switch (num_parents) {
    case 1:
      El::LockedView(get_error_signals(), get_prev_error_signals());
      break;
    case 2:
      {
        const auto& local_input0 = get_local_prev_activations(0);
        const auto& local_input1 = get_local_prev_activations(1);
        auto& local_gradient_wrt_input0 = get_local_error_signals(0);
        auto& local_gradient_wrt_input1 = get_local_error_signals(1);
	LBANN_OMP_TASKLOOP_COLLAPSE2
        for (int col = 0; col < local_width; ++col) {
          for (int row = 0; row < local_height; ++row) {
            const auto& x0 = local_input0(row, col);
            const auto& x1 = local_input1(row, col);
            const auto& dy = local_gradient_wrt_output(row, col);
            auto& dx0 = local_gradient_wrt_input0(row, col);
            auto& dx1 = local_gradient_wrt_input1(row, col);
            if (x0 < x1) {
              dx0 = dy;
              dx1 = zero;
            } else if (x0 > x1) {
              dx0 = zero;
              dx1 = dy;
            } else {
              dx0 = dy / 2;
              dx1 = dy / 2;
            }
          }
        }
      }
      break;
    default:
      LBANN_OMP_TASKLOOP_COLLAPSE2
      for (int col = 0; col < local_width; ++col) {
        for (int row = 0; row < local_height; ++row) {
          const auto& dy = local_gradient_wrt_output(row, col);

          // Find minimum input
          int min_index = 0;
          int min_value = get_local_activations(0)(row, col);
          for (int i = 1; i < num_parents; ++i) {
            const auto& current_value = get_local_activations(i)(row, col);
            if (current_value < min_value) {
              min_index = i;
              min_value = current_value;
            }
          }

          // Output error signal to minimum input
          for (int i = 0; i < num_parents; ++i) {
            auto& dx = get_local_error_signals(i)(row, col);
            dx = (i == min_index) ? dy : zero;
          }

        }
      }
    }

  }

};

} // namespace lbann

#endif // LBANN_LAYER_MIN_HPP_INCLUDED
