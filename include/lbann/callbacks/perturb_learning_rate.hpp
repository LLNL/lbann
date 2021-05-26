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

#ifndef LBANN_CALLBACKS_CALLBACK_PERTURB_LEARNING_RATE_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_PERTURB_LEARNING_RATE_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/optimizers/data_type_optimizer.hpp"

#include <set>

namespace lbann {
namespace callback {

/** @brief Hyperparameter exploration of optimizer learning rate.
 *
 *  Goes through optimizers in a model and perturbs
 *  the learning rate. Current implementation supports 
 *  random perturbation, performed in log space.
 *  More precisely, random values are drawn from normal distributions
 *  (with user-provided standard deviations) and added to
 *  @f$\log(\text{learning rate})@f$.
 *  Currently implementation can be extended to support other 
 *  exploration techniques e.g., Resample 
 */
class perturb_learning_rate : public callback_base {
public:

  /** @param learning_rate_factor   Standard deviation of learning rate
   *                                perturbation (in log space).
   *                                perturbation (in log space).
   *  @param perturb_during_training    Whether to periodically perturb
   *                                    learning rate during training
   *                                    or to only perturb once during
   *                                    setup.
   *  @param batch_interval Number of training mini-batch steps between
   *                        perturbations. Only used if
   *                        @c perturb_during_training is @c true.
   *  @param weights_names  Names of weights with optimizers. If
   *                        empty, all optimizers in the model are
   *                        perturbed.
   */
  perturb_learning_rate(DataType learning_rate_factor,
               bool perturb_during_training = false,
               El::Int batch_interval = 1,
               std::set<std::string> weights_names
               = std::set<std::string>());
  perturb_learning_rate* copy() const override { return new perturb_learning_rate(*this); }
  std::string name() const override { return "perturb optimizer learning rate"; }

  void setup(model* m) override;
  void on_batch_begin(model* m) override;

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar);

  ///@}

private:

  friend class cereal::access;
  perturb_learning_rate();

  /** Standard deviation of learning rate perturbation.
   *
   *  In log space.
   */
  DataType m_learning_rate_factor;

  /** Whether to periodically perturb during training.
   *
   *  If false, only perturb once during setup.
   */
  bool m_perturb_during_training;

  /** Optimizers for these weights will be perturbed.
   *
   *  If empty, all optimizers in the model will be perturbed.
   */
  std::set<std::string> m_weights_names;

  /** Perturb optimizers in model. */
  void perturb(model& m) const;
  /** Perturb optimizer learning rate. */
  void perturb(lbann_comm& comm, data_type_optimizer<DataType>& opt) const;

};

// Builder function
std::unique_ptr<callback_base>
build_perturb_learning_rate_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_PERTURB_LEARNING_RATE_HPP_INCLUDED
