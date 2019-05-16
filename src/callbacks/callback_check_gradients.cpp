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

#include "lbann/callbacks/callback_check_gradients.hpp"

namespace lbann {

lbann_callback_check_gradients
  ::lbann_callback_check_gradients(DataType step_size,
                                   bool verbose,
                                   bool error_on_failure)
  : m_step_size(step_size),
    m_verbose(verbose),
    m_error_on_failure(error_on_failure) {}

void lbann_callback_check_gradients::on_test_begin(model *m) {

  // Get model members
  lbann_comm *comm = m->get_comm();
  const std::vector<Layer*>& layers = m->get_layers();

  // Initialize network for testing
  for (auto&& w : m->get_weights()) {
    auto&& opt = w->get_optimizer();
    if (opt != nullptr) { opt->clear_gradient(); }
  }
  layers[0]->forward_prop();

  // Compute objective function
  const DataType objective = compute_objective_function(m);

  // Choose finite difference step
  // Note: Consider a central difference scheme:
  //   f'(x) ~ ( - f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h) ) / 12h
  // By Taylor's theorem, the truncation error is bounded by
  //   E_trunc <= | f'''''(xi) | / 18 * h^4
  // Assuming f can be computed to a relative accuracy of epsilon,
  //   E_fl <= epsilon * | f(chi) | / h
  // For simplicity, we assume f(chi) ~ f(x), and | f'''''(xi) | ~ 1.
  // If step size is not specified, then we choose h so that
  //   E_fl <= sqrt(epsilon)
  const DataType epsilon = std::pow(std::numeric_limits<DataType>::epsilon(), 0.9);
  DataType step_size = m_step_size;
  if (m_step_size <= DataType(0)) {
    step_size = std::fabs(objective) * std::sqrt(epsilon);
  }
  DataType expected_error = (epsilon * objective / step_size
                             + std::pow(step_size, 4) / 18);
  expected_error = std::pow(expected_error, 0.9);

  // Compute gradients
  m->get_objective_function()->differentiate();
  m->get_objective_function()->compute_weight_regularization();
  for (int l = layers.size() - 1; l > 0; --l) {
    layers[l]->back_prop();
  }

  // Print objective function value
  if (comm->am_world_master()) {
    std::cout << "--------------------------------------------------------------------------------" << std::endl
              << "Gradient checking..." << std::endl
              << "  Objective function value = " << objective << std::endl
              << "  Step size                = " << step_size << std::endl
              << "  Expected gradient error  = " << expected_error << std::endl;
  }

  for (weights *w : m->get_weights()) {
    if (w->get_optimizer() == nullptr) {
      continue;
    }
    if (comm->am_world_master()) {
      std::cout << "Checking " << w->get_name() << std::endl;
    }

    // Get weights matrix and gradient
    const AbsDistMat& weights_matrix = w->get_values();
    const AbsDistMat& gradient = w->get_optimizer()->get_gradient();

    // Iterate through weights matrix entries
    for (El::Int col = 0; col < weights_matrix.Width(); ++col) {
      for (El::Int row = 0; row < weights_matrix.Height(); ++row) {
        const bool weight_is_local = weights_matrix.IsLocal(row, col);
        const El::Int local_row = (weight_is_local ?
                                   weights_matrix.LocalRow(row) :
                                   0);
        const El::Int local_col = (weight_is_local ?
                                   weights_matrix.LocalCol(col) :
                                   0);
        const DataType initial_weight = (weight_is_local ?
                                         weights_matrix.GetLocal(local_row,
                                                                 local_col) :
                                         DataType(0));

        // Compute objective function values
        // Note: matrix entry is reset after computing objective
        // function values
        w->set_value(initial_weight + 2 * step_size, row, col);
        const DataType f_2h = compute_objective_function(m);
        w->set_value(initial_weight + step_size, row, col);
        const DataType f_h = compute_objective_function(m);
        w->set_value(initial_weight - step_size, row, col);
        const DataType f_nh = compute_objective_function(m);
        w->set_value(initial_weight - 2 * step_size, row, col);
        const DataType f_n2h = compute_objective_function(m);
        w->set_value(initial_weight, row, col);

        // Compute relative error in gradient.
        // Note: only weight owner participates
        if (weight_is_local && weights_matrix.RedundantRank() == 0) {
          const DataType analytical_gradient
            = gradient.GetLocal(local_row, local_col);
          const DataType numerical_gradient
            = (- f_2h + 8 * f_h - 8 * f_nh + f_n2h) / (12 * step_size);
          const DataType error = std::fabs(analytical_gradient - numerical_gradient);
          auto relative_error = DataType(0);
          if (error != DataType(0)) {
            relative_error = error / std::max(std::fabs(analytical_gradient),
                                              std::fabs(numerical_gradient));
          }

          // Print warning if relative error is large
          if (error > expected_error || std::isnan(error) || std::isinf(error)) {
            std::cout << "  GRADIENT ERROR: " << w->get_name() << ", "
                      << "entry (" << row << "," << col << ")" << std::endl;
            std::cout << "    Weight              = " << initial_weight << std::endl
                      << "    Analytical gradient = " << analytical_gradient << std::endl
                      << "    Numerical gradient  = " << numerical_gradient << std::endl
                      << "    Error               = " << error << std::endl
                      << "    Relative error      = " << relative_error << std::endl;
            if (m_error_on_failure) {
              throw lbann_exception("callback_check_gradients: found large error in gradient");
            }
          } else if (m_verbose) {
            std::cout << "  " << w->get_name() << ", "
                      << "entry (" << row << "," << col << ")" << std::endl;
            std::cout << "    Weight              = " << initial_weight << std::endl
                      << "    Analytical gradient = " << analytical_gradient << std::endl
                      << "    Numerical gradient  = " << numerical_gradient << std::endl
                      << "    Error               = " << error << std::endl
                      << "    Relative error      = " << relative_error << std::endl;
          }
        }

      }
    }

  }

  if (comm->am_world_master()) {
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
  }

}

DataType lbann_callback_check_gradients::compute_objective_function(model *m) {
  const std::vector<Layer*>& layers = m->get_layers();
  objective_function* obj_fn = m->get_objective_function();
  for (size_t l = 1; l < layers.size(); l++) {
    layers[l]->forward_prop();
  }
  obj_fn->start_evaluation(m->get_execution_mode(),
                           m->get_current_mini_batch_size());
  return obj_fn->finish_evaluation(m->get_execution_mode(),
                                   m->get_current_mini_batch_size());
}

}  // namespace lbann
