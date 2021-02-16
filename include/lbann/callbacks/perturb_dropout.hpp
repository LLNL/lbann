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

#ifndef LBANN_CALLBACKS_CALLBACK_PERTURB_DROPOUT_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_PERTURB_DROPOUT_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/layers/regularizers/dropout.hpp"
#include <cereal/types/set.hpp>
#include <set>

namespace lbann {
namespace callback {

/** @brief Hyperparameter exploration with dropouts.
 *
 *  Goes through the dropout layers in a model and perturbs keep probability
 */
class perturb_dropout : public callback_base {
public:

  /** @param keep_prob_factor   Standard deviation of learning rate
   *                                perturbation (in log space).
   *  @param layer_names  Names of layers with dropout keep prob to perturb. If
   *                        empty, all dropout layers  in the model are
   *                        perturbed.
   */
  perturb_dropout(EvalType keep_prob_factor,
                              std::set<std::string> layer_names
                              = std::set<std::string>());
  perturb_dropout* copy() const override { return new perturb_dropout(*this); }
  std::string name() const override { return "perturb dropout"; }

  void setup(model* m) override;

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar) {
    ar(::cereal::make_nvp(
         "BaseCallback",
         ::cereal::base_class<callback_base>(this)),
       CEREAL_NVP(m_keep_prob_factor),
       CEREAL_NVP(m_layer_names));
  }

  ///@}

private:

  friend class cereal::access;
  perturb_dropout();

  /** Standard deviation of keep probability  perturbation.
   *
   *  In log space.
   */
  EvalType m_keep_prob_factor;

  /** Keep prob for these layers will be perturbed.
   *
   *  If empty, all dropout layers  in the model will be perturbed.
   */
  std::set<std::string> m_layer_names;

  template <typename TensorDataType, data_layout T_layout, El::Device Dev>
  dropout<TensorDataType, T_layout, Dev>* get_dropout_layer(Layer* l);

  /** Perturb dropout keep prob in model. */
  void perturb(model& m);

};

// Builder function
std::unique_ptr<callback_base>
build_perturb_dropout_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_PERTURB_DROPOUT_HPP_INCLUDED
