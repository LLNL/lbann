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
  min_layer(lbann_comm *comm,
            cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm) {

    /// @todo Implement
    static_assert(Dev == El::Device::CPU,
                  "min layer currently only supports CPU");

    // Min layer has no limit on parents
    m_expected_num_parent_layers = -1;

  #ifdef LBANN_HAS_CUDNN
    // Initialize GPU if available
    this->m_cudnn = cudnn;
  #endif // LBANN_HAS_CUDNN

  }

  min_layer* copy() const override { return new min_layer(*this); }
  std::string get_type() const override { return "min"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:

  void setup_dims() override {
    transform_layer::setup_dims();
    for (const auto& parent : this->m_parent_layers) {
      const auto& parent_dims = parent->fp_output_dims(this);
      if (m_neuron_dims != parent_dims) {
        std::stringstream err;
        err << "layer " << get_name() << " expects inputs with "
            << "dimensions ";
        for (size_t i = 0; i < m_neuron_dims.size(); ++i) {
          err << (i > 0 ? "x" : "") << m_neuron_dims[i];
        }
        err << ", but layer " << parent->get_name() << " outputs with "
            << "dimensions ";
        for (size_t i = 0; i < parent_dims.size(); ++i) {
          err << (i > 0 ? "x" : "") << parent_dims[i];
        }
        LBANN_ERROR(err.str());
      }
    }
  }

  void fp_compute() override {
    const int num_parents = get_num_parents();
    auto& output = get_activations();
    switch (num_parents) {
    case 0: El::Zero(output); break;
    case 1: El::LockedView(output, get_prev_activations()); break;
    default:

      // Local matrices
      const auto& local_input0 = get_local_prev_activations(0);
      const auto& local_input1 = get_local_prev_activations(1);
      auto& local_output = output.Matrix();
      const int local_height = local_output.Height();
      const int local_width = local_output.Width();

      // Minimum of first two inputs
      #pragma omp parallel for collapse(2)
      for (int col = 0; col < local_width; ++col) {
        for (int row = 0; row < local_height; ++row) {
          local_output(row, col) = std::min(local_input0(row, col),
                                            local_input1(row, col));
        }
      }

      // Case with more than 2 parents
      for (int i = 2; i < num_parents; ++i) {
        const auto& local_input = get_local_prev_activations(i);
        #pragma omp parallel for collapse(2)
        for (int col = 0; col < local_width; ++col) {
          for (int row = 0; row < local_height; ++row) {
            const auto& x = local_input(row, col);
            auto& y = local_output(row, col);
            if (x < y) { y = x; }
          }
        }
      }

    }
  }

  void bp_compute() override {

    const int num_parents = get_num_parents();

    const auto& gradient_wrt_output = get_prev_error_signals();
    const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
    const int local_height = local_gradient_wrt_output.Height();
    const int local_width = local_gradient_wrt_output.Width();

    switch (num_parents) {
    case 0: break;
    case 1:
      El::LockedView(get_error_signals(), gradient_wrt_output);
      break;
    case 2:
      {
        const auto& local_input0 = get_local_prev_activations(0);
        const auto& local_input1 = get_local_prev_activations(1);
        auto& local_gradient_wrt_input0 = get_local_error_signals(0);
        auto& local_gradient_wrt_input1 = get_local_error_signals(1);
        #pragma omp parallel for collapse(2)
        for (int col = 0; col < local_width; ++col) {
          for (int row = 0; row < local_height; ++row) {
            const auto& dy = local_gradient_wrt_output(row, col);
            if (local_input0(row, col) <= local_input1(row, col)) {
              local_gradient_wrt_input0(row, col) += dy;
              local_gradient_wrt_input1(row, col) += DataType(0);
            } else {
              local_gradient_wrt_input0(row, col) += DataType(0);
              local_gradient_wrt_input1(row, col) += dy;
            }
          }
        }
      }
      break;
    default:
      #pragma omp parallel for collapse(2)
      for (int col = 0; col < local_width; ++col) {
        for (int row = 0; row < local_height; ++row) {

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
            get_local_error_signals(i)(row, col)
              += (i == min_index ?
                  local_gradient_wrt_output(row, col) :
                  DataType(0));
          }

        }
      }
    }

  }

};

} // namespace lbann

#endif // LBANN_LAYER_MAX_HPP_INCLUDED
