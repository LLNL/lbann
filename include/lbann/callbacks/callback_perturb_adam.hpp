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

#ifndef LBANN_CALLBACKS_CALLBACK_PERTURB_ADAM_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_PERTURB_ADAM_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/optimizers/adam.hpp"
#include <set>

namespace lbann {

/** @brief Hyperparameter exploration with Adam optimizers.
 *
 *  Goes through the Adam optimizers in a model and perturbs four
 *  hyperparameters: the learning rate, @f$\beta_1@f$, @f$\beta_2@f$,
 *  and @f$\epsilon@f$. Since these hyperparameters can range over
 *  orders of magnitude, the perturbations are performed in log space.
 *  More precisely, random values are drawn from normal distributions
 *  (with user-provided standard deviations) and added to
 *  @f$\log(\text{learning rate})@f$, @f$\log(1-\beta_1)@f$,
 *  @f$\log(1-\beta_2)@f$, and @f$\log\epsilon@f$.
 */
class lbann_callback_perturb_adam : public lbann_callback {
public:

  /** @param learning_rate_factor   Standard deviation of learning rate
   *                                perturbation (in log space).
   *  @param beta1_factor           Standard deviation of @f$\beta_1@f$
   *                                perturbation (in log space).
   *  @param beta2_factor           Standard deviation of @f$\beta_2@f$
   *                                perturbation (in log space).
   *  @param eps_factor             Standard deviation of @f$\epsilon@f$
   *                                perturbation (in log space).
   *  @param perturb_during_training    Whether to periodically perturb
   *                                    hyperparameters during training
   *                                    or to only perturb once during
   *                                    setup.
   *  @param batch_interval Number of training mini-batch steps between
   *                        perturbations. Only used if
   *                        @c perturb_during_training is @c true.
   *  @param weights_names  Names of weights with Adam optimizers. If
   *                        empty, all Adam optimizers in the model are
   *                        perturbed.
   */
  lbann_callback_perturb_adam(DataType learning_rate_factor,
                              DataType beta1_factor,
                              DataType beta2_factor,
                              DataType eps_factor = 0,
                              bool perturb_during_training = false,
                              El::Int batch_interval = 1,
                              std::set<std::string> weights_names
                              = std::set<std::string>());
  lbann_callback_perturb_adam* copy() const override { return new lbann_callback_perturb_adam(*this); }
  std::string name() const override { return "perturb Adam"; }

  void setup(model* m) override;
  void on_batch_begin(model* m) override;

private:

  /** Standard deviation of learning rate perturbation.
   *
   *  In log space.
   */
  DataType m_learning_rate_factor;
  /** Standard deviation of @f$\beta_1@f$ perturbation.
   *
   *  In log space.
   */
  DataType m_beta1_factor;
  /** Standard deviation of @f$\beta_2@f$ perturbation.
   *
   *  In log space.
   */
  DataType m_beta2_factor;
  /** Standard deviation of @f$\epsilon@f$ perturbation.
   *
   *  In log space.
   */
  DataType m_eps_factor;

  /** Whether to periodically perturb during training.
   *
   *  If false, only perturb once during setup.
   */
  bool m_perturb_during_training;

  /** Optimizers for these weights will be perturbed.
   *
   *  If empty, all Adam optimizers in the model will be perturbed.
   */
  std::set<std::string> m_weights_names;

  /** Perturb Adam optimizers in model. */
  void perturb(model& m) const;
  /** Perturb Adam optimizer hyperparameters. */
  void perturb(lbann_comm& comm, adam& m) const;

};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_perturb_adam_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_PERTURB_ADAM_HPP_INCLUDED
