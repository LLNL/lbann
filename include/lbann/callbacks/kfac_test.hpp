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
//
// callback_kfac_test .hpp .cpp - Callbacks for the K-FAC method
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_KFAC_TEST_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_KFAC_TEST_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

// Add the damping value to the diagonal elements of A.
template <typename TensorDataType>
void kfac_test_add_to_diagonal(
    TensorDataType * __restrict__ A,
    const size_t height,
    const TensorDataType damping);

// Fill the upper trianglar with the lower trianglar.
template <typename TensorDataType>
void kfac_test_fill_upper_tri(
    TensorDataType * __restrict__ A,
    const size_t height);

/** Callback hooks for the K-FAC method. */
class kfac_test : public callback_base {
 public:

  /** Constructor.
   */
  kfac_test(double damping_0, double damping_target,
            double damping_warmup_steps,
            bool print_time, bool print_matrix,
            bool print_matrix_summary)
      : callback_base(),
        m_damping_0(damping_0), m_damping_target(damping_target),
        m_damping_warmup_steps(damping_warmup_steps),
        m_print_time(print_time), m_print_matrix(print_matrix),
        m_print_matrix_summary(print_matrix_summary) {
    m_damping = m_damping_0;
  }
  kfac_test(const kfac_test&) = default;
  kfac_test& operator=(const kfac_test&) = default;
  kfac_test* copy() const override { return new kfac_test(*this); }
  void setup(model *m) override;
  void on_backward_prop_end(model *m) override;
  void on_epoch_end(model *m) override;
  void on_backward_prop_end(model *m, Layer *l) override;
  std::string name() const override { return "K-FAC test"; }

  /** @brief The default parameters of a Tikhonov damping technique. */
  constexpr static const double damping_0_default = 3e-2;
  constexpr static const double damping_target_default = 1e-4;
  constexpr static const double damping_warmup_steps_default = 100;

 private:

  /** @brief Parameters of a Tikhonov damping technique. */
  const double m_damping_0, m_damping_target, m_damping_warmup_steps;

  /** @brief Knobs to print information for debugging. */
  const bool m_print_time, m_print_matrix, m_print_matrix_summary;
  double m_damping;

};

// Builder function
std::unique_ptr<callback_base>
build_kfac_test_callback_from_pbuf(
    const google::protobuf::Message&,std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_TEST_HPP_INCLUDED
