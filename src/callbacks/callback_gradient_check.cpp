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
  const DataType objective = m->m_obj_fn->get_value();
  m->m_obj_fn->reset_statistics();

  // Choose finite difference step
  // Note: Consider a central difference scheme:
  //   f'(x) ~ ( f(x+h) - f(x-h) ) / 2h
  // By Taylor's theorem, the truncation error is bounded by
  //   E_trunc <= | f'''(xi) | / 6 * h^2
  // By basic numerical analysis, the floating point error is bounded by
  //   E_fl <= eps * | f(chi) | / h
  // The bound E = E_trunc + E_fl is minimized with
  //   h = cbrt( 3 * epsilon * | f(chi) | / | f'''(xi) | )
  // For simplicity, we assume f(chi) ~ f(x), and | f'''(xi) | ~ 1.
  const DataType epsilon = std::numeric_limits<DataType>::epsilon();
  const DataType effective_objective = (objective != DataType(0) ?
                                        objective : DataType(1));
  const DataType step = std::cbrt(3 * epsilon * effective_objective);
  DataType expected_error = (epsilon * effective_objective / step
                             + step * step / 6);
  expected_error = std::pow(expected_error, 0.75);

  // Compute gradients
  for (size_t l = layers.size(); l-- > 0u;) {
    layers[l]->back_prop();
  }

  // Print objective function value
  if (comm->am_world_master()) {
    std::cout << "--------------------------------------------------------------------------------" << std::endl
              << "Gradient checking..." << std::endl
              << "  Objective function value = " << objective << std::endl
              << "  Expected gradient error  = " << expected_error << std::endl;
  }

  // Iterate through layers
  for (size_t layer_index = 1; layer_index < layers.size() - 1; ++layer_index) {

    // Check that current layer is a learning layer
    learning* layer = dynamic_cast<learning*>(layers[layer_index]);
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
        const DataType analytical_gradient = weights_gradient.Get(row, col);
        const DataType numerical_gradient = (objective_plus - objective_minus) / (2 * step);
        const DataType error = std::fabs(analytical_gradient - numerical_gradient);
        DataType relative_error = DataType(0);
        if (error != DataType(0)) {
          relative_error = error / std::max(std::fabs(analytical_gradient),
                                            std::fabs(numerical_gradient));
        }
        
        // Print warning if relative error is large
        if (error > expected_error && comm->am_world_master()) {
          std::cout << "  Gradient error in layer " << layer_index << ", "
                    << "entry (" << row << "," << col << ")" << std::endl;
          std::cout << "    Weight              = " << initial_weight << std::endl
                    << "    Step                = " << step << std::endl
                    << "    Analytical gradient = " << analytical_gradient << std::endl
                    << "    Numerical gradient  = " << numerical_gradient << std::endl
                    << "    Error               = " << error << std::endl
                    << "    Relative error      = " << relative_error << std::endl;
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
