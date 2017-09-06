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
//
// lbann_callback_gradient_check .hpp .cpp - Callback hooks for gradient check
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_gradient_check.hpp"

namespace lbann {

lbann_callback_gradient_check::lbann_callback_gradient_check(DataType max_error)
  : m_max_error(max_error) {}

void lbann_callback_gradient_check::on_test_begin(model *m) {

  // Get model members
  lbann_comm *comm = m->get_comm();
  std::vector<Layer*>& layers = m->get_layers();

  // Initialize network for testing
  for (size_t l = 0; l < layers.size(); ++l) {
    layers[l]->set_execution_mode(execution_mode::testing);
  }
  m->m_obj_fn->reset_statistics();

  // Compute objective function
  for (size_t l = 0; l < layers.size(); l++) {
    layers[l]->forward_prop();
  }
  const DataType objective_function_value = m->m_obj_fn->get_value();
  m->m_obj_fn->reset_statistics();

  // Compute gradients
  for (size_t l = layers.size(); l-- > 0u;) {
    layers[l]->back_prop();
  }

  // Print objective function value
  if (comm->am_world_master()) {
    std::cout << "--------------------------------------------------------------------------------" << std::endl
              << "Gradient checking..." << std::endl
              << "  Objective function value = " << objective_function_value << std::endl;
  }

  // Iterate through layers
  for (size_t l = 1; l < layers.size() - 1; ++l) {

    // Check that current layer is a learning layer
    learning* layer = dynamic_cast<learning*>(layers[l]);
    if (layer == nullptr) {
      continue;
    }

    // Get weights and gradients
    auto& weights = layer->get_weights();
    auto& weights_gradient = layer->get_weights_gradient();

    // Iterate through weights in current layer
    for (El::Int col = 0; col < weights.Width(); ++col) {
      for (El::Int row = 0; row < weights.Height(); ++row) {
        const DataType initial_weight = weights.Get(row, col);

        // Choose finite difference step
        DataType step = std::sqrt(std::numeric_limits<DataType>::epsilon());
        if(initial_weight != DataType(0)) {
          step *= std::fabs(initial_weight);
        }

        // Compute objective function with positive step
        weights.Set(row, col, initial_weight + step);
        for (size_t l = 1; l < layers.size(); l++) {
          layers[l]->forward_prop();
        }
        const DataType objective_plus = m->m_obj_fn->get_value();
        m->m_obj_fn->reset_statistics();

        // Compute objective function with negative step
        weights.Set(row, col, initial_weight - step);
        for (size_t l = 1; l < layers.size(); l++) {
          layers[l]->forward_prop();
        }
        const DataType objective_minus = m->m_obj_fn->get_value();
        m->m_obj_fn->reset_statistics();
        
        // Compute relative error in gradient
        const DataType backprop_gradient = weights_gradient.Get(row, col);
        const DataType numerical_gradient = (objective_plus - objective_minus) / (2 * step);
        const DataType error = std::fabs(backprop_gradient - numerical_gradient);
        const DataType relative_error = error / std::max(std::fabs(backprop_gradient),
                                                         std::fabs(numerical_gradient));
        
        // Print warning if relative error is large
        if (relative_error > m_max_error
            && comm->am_world_master()) {
          std::cout << "  Gradient error in layer " << l << ", "
                    << "entry (" << row << "," << col << ")" << std::endl;
          std::cout << "    Backprop gradient  = " << backprop_gradient << std::endl
                    << "    Numerical gradient = " << numerical_gradient << std::endl
                    << "    Error              = " << error << std::endl
                    << "    Relative error     = " << relative_error << std::endl;
        }

        // Reset weight
        weights.Set(row, col, initial_weight);

      }
    }

  }

  if (comm->am_world_master()) {
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
  }

}

}  // namespace lbann
