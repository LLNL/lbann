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

#ifndef LBANN_LAYER_REGULARIZER_SELU_DROPOUT_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_SELU_DROPOUT_HPP_INCLUDED

#include "lbann/layers/regularizers/regularizer.hpp"

namespace lbann {

/**
 * SELU dropout: alpha-scaled dropout for use with SELU activations.
 * See: Klambauer et al. "Self-Normalizing Neural Networks", 2017.
 * This makes the same default assumptions as our SELU activations.
 * The paper recommends a default dropout rate of 0.05 (keep 0.95).
 */
template <data_layout T_layout, El::Device Dev>
class selu_dropout : public regularizer_layer {
 public:
  /** Keep units with probabiliy keep_prob. */
  selu_dropout(lbann_comm *comm,
               float keep_prob=0.95f,
               DataType alpha = DataType(1.6732632423543772848170429916717),
               DataType scale = DataType(1.0507009873554804934193349852946)) :
    regularizer_layer(comm),
    m_keep_prob(keep_prob),
    m_mask(nullptr) {
#ifdef LBANN_DETERMINISTIC
    throw lbann_exception("selu_dropout: deterministic dropout not supported");
#endif
    // Compute alpha' and the affine transform.
    m_alpha_prime = -scale*alpha;
    m_a = keep_prob +
      m_alpha_prime*m_alpha_prime*keep_prob*(DataType(1) - keep_prob);
    m_a = DataType(1) / std::sqrt(m_a);
    m_b = -m_a * m_alpha_prime*(DataType(1) - keep_prob);
  }

  selu_dropout(const selu_dropout& other) :
    regularizer_layer(other),
    m_alpha_prime(other.m_alpha_prime),
    m_a(other.m_a),
    m_b(other.m_b),
    m_keep_prob(other.m_keep_prob),
    m_mask(other.m_mask) {
    if (m_mask != nullptr) { m_mask = m_mask->Copy(); }
  }

  selu_dropout& operator=(const selu_dropout& other) {
    regularizer_layer::operator=(other);
    m_alpha_prime = other.m_alpha_prime;
    m_a = other.m_a;
    m_b = other.m_b;
    m_keep_prob = other.m_keep_prob;
    if (m_mask != nullptr) { delete m_mask; }
    m_mask = other.m_mask;
    if (m_mask != nullptr) { m_mask = m_mask->Copy(); }
    return *this;
  }

  ~selu_dropout() override {
    if (m_mask != nullptr) { delete m_mask; }
  }

  selu_dropout* copy() const override { return new selu_dropout(*this); }

  std::string get_type() const override { return "selu dropout"; }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  void setup_matrices(const El::Grid& grid) override {
    regularizer_layer::setup_matrices(grid);
    if (m_mask != nullptr) { delete m_mask; }
    m_mask = get_activations().Copy();
  }

 protected:
  /** Drop out units in forward propagation. */
  void fp_compute() override {
    if (this->m_model->get_execution_mode() != execution_mode::training ||
        m_keep_prob < 0.0f) {
      // Do nothing if dropout is disabled
      El::Copy(get_prev_activations(), get_activations());
    } else {

      const auto *input_acts = &get_prev_activations();
      const IntType height = input_acts->Height();
      const IntType width = input_acts->Width();
      const IntType local_height = input_acts->LocalHeight();
      const IntType local_width = input_acts->LocalWidth();

      const auto& local_input_acts = input_acts->LockedMatrix();
      Mat& local_output_acts = get_local_activations();
      Mat& local_mask = m_mask->Matrix();

      // Construct and apply mask and the affine transform.
      // TODO: Optimize.
      El::Bernoulli(*m_mask, height, width, m_keep_prob);
      for (IntType col = 0; col < local_width; ++col) {
        for (IntType row = 0; row < local_height; ++row) {
          local_output_acts(row, col) = m_a *
            (local_input_acts(row, col)*local_mask(row, col) +
             m_alpha_prime*(1 - local_mask(row, col))) + m_b;
        }
      }

    }
  }

  /** Adjust gradients for dropout in backprop. */
  void bp_compute() override {
    if (this->m_model->get_execution_mode() != execution_mode::training
        || m_keep_prob < 0.0f) {
      El::Copy(get_prev_error_signals(), get_error_signals());
    } else {

      const auto& local_prev_error_signal = get_local_prev_error_signals();
      Mat& local_error_signal = get_local_error_signals();
      Mat& local_mask = m_mask->Matrix();
      const IntType local_height = local_prev_error_signal.Height();
      const IntType local_width = local_prev_error_signal.Width();
      // Reweight with the affine scale factor and the dropout mask.
      for (IntType col = 0; col < local_width; ++col) {
        for (IntType row = 0; row < local_height; ++row) {
          local_error_signal(row, col) =
            m_a * local_prev_error_signal(row, col) * local_mask(row, col);
        }
      }

    }
  }

 private:
  /** Alpha prime, the low-variance saturation point. */
  DataType m_alpha_prime;
  /** Affine scaling parameter to keep mean/variance at desired value. */
  DataType m_a;
  /** Affine additive parameter to keep mean/variance at desired value. */
  DataType m_b;
  /** Probability of keeping each unit. */
  float m_keep_prob;
  /** Current dropout mask (a scaled Bernoulli random matrix). */
  AbsDistMat *m_mask;
};

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_SELU_DROPOUT_HPP_INCLUDED
