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
#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/data_readers/data_reader.hpp"

#include "callbacks.pb.h"

namespace lbann {

namespace {

/** @details Forward prop is applied to all layers, except input
 *  layers. It is assumed that input layers have already loaded data.
 */
DataType compute_objective_function(model& m) {

  // Forward prop, skipping input layers
  for (auto&& l : m.get_layers()) {
    if (dynamic_cast<generic_input_layer*>(l) == nullptr) {
      l->forward_prop();
    }
  }

  // Get objective function value
  auto&& obj = m.get_objective_function();
  const auto mode = m.get_execution_mode();
  const auto mini_batch_size = m.get_current_mini_batch_size();
  obj->start_evaluation(mode, mini_batch_size);
  return obj->finish_evaluation(mode, mini_batch_size);

}

} // namespace

lbann_callback_check_gradients
  ::lbann_callback_check_gradients(DataType step_size,
                                   bool verbose,
                                   bool error_on_failure)
  : m_step_size(step_size),
    m_verbose(verbose),
    m_error_on_failure(error_on_failure) {}

void lbann_callback_check_gradients::on_test_end(model *m) {

  // Get objects from model
  lbann_comm *comm = m->get_comm();
  auto mode = m->get_execution_mode();
  const auto& layers = m->get_layers();

  // Reset statistics and gradients
  m->get_objective_function()->reset_statistics(mode);
  for (auto&& met : m->get_metrics()) {
    met->reset_statistics(mode);
  }
  for (auto&& w : m->get_weights()) {
    auto&& opt = w->get_optimizer();
    if (opt != nullptr) { opt->clear_gradient(); }
  }

  // Load data in input layers
  for (auto&& l : m->get_layers()) {
    if (dynamic_cast<generic_input_layer*>(l) != nullptr) {
      l->forward_prop();
    }
  }

  // Compute objective function
  const DataType objective = compute_objective_function(*m);

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
    std::cout << "----------------------------------------------------------------\n"
              << "Gradient checking...\n"
              << "  Objective function value = " << objective << "\n"
              << "  Step size                = " << step_size << "\n"
              << "  Expected gradient error  = " << expected_error << "\n";
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
        const DataType f_2h = compute_objective_function(*m);
        w->set_value(initial_weight + step_size, row, col);
        const DataType f_h = compute_objective_function(*m);
        w->set_value(initial_weight - step_size, row, col);
        const DataType f_nh = compute_objective_function(*m);
        w->set_value(initial_weight - 2 * step_size, row, col);
        const DataType f_n2h = compute_objective_function(*m);
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
    std::cout << "----------------------------------------------------------------\n";
  }

  // Clean up
  /// @todo tym: I'm not sure if data readers are properly reset
  for (auto&& l : m->get_layers()) {
    auto&& input = dynamic_cast<generic_input_layer*>(l);
    if (input != nullptr) {
      auto&& reader = input->get_data_reader(mode);
      reader->set_initial_position();
    }
  }
  m->get_objective_function()->reset_statistics(mode);
  for (auto&& met : m->get_metrics()) {
    met->reset_statistics(mode);
  }

}

// Builder function
std::unique_ptr<lbann_callback>
build_callback_check_gradients_from_pbuf(
  const google::protobuf::Message& proto_msg, lbann_summary*) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackCheckGradients&>(proto_msg);
  return make_unique<lbann_callback_check_gradients>(params.step_size(),
                                                     params.verbose(),
                                                     params.error_on_failure());
}

} // namespace lbann
