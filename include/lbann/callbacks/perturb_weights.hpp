////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_CALLBACKS_CALLBACK_PERTURB_WEIGHTS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_PERTURB_WEIGHTS_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/weights/weights.hpp"

namespace lbann {
namespace callback {

/** @brief Perturb values in a weights tensor.
 *
 *  Each entry of the weights tensor has a probability of being
 *  perturbed by a normal random number. The resulting values are
 *  clamped within a range.
 */
class perturb_weights : public callback_base {
public:

  /**
   *  @param batch_interval Number of training mini-batch steps
   *                        between perturbations
   *  @param output_name    Name of weights being perturbed
   *  @param upper          Upper bound for weights values
   *  @param lower          Lower bound for weights values
   *  @param scale          Standard deviation of normal perturbations
   *  @param perturb_probability    Probability of applying
   *                        perturbation to a given weights value
   */
  perturb_weights(EvalType upper, EvalType lower, EvalType scale, EvalType perturb_probability,
		  std::string output_name,
                  El::Int batch_interval = 1);

  perturb_weights* copy() const override { return new perturb_weights(*this); }
  std::string name() const override { return "perturb weights"; }

  void setup(model* m) override;
  void on_batch_begin(model* m) override;

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar);

  ///@}

private:

  friend class cereal::access;
  perturb_weights();

  /// @brief Name of weights being perturbed
  std::string m_output_name;

  /// @brief Upper bound for weights values
  EvalType m_upper;
  /// @brief Lower bound for weights values
  EvalType m_lower;
  /// @brief Standard deviation of normal perturbations
  EvalType m_scale;
  /// @brief Probability of applying perturbation to a given value
  EvalType m_perturb_probability;

  void perturb(model& m);

};

// Builder function
std::unique_ptr<callback_base>
build_perturb_weights_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_PERTURB_WEIGHTS_HPP_INCLUDED
