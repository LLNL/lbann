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

#ifndef LBANN_CALLBACKS_MIXUP_HPP
#define LBANN_CALLBACKS_MIXUP_HPP

#include "lbann/callbacks/callback.hpp"

#include <cereal/types/unordered_set.hpp>
#include <unordered_set>
#include <string>

namespace lbann {
namespace callback {

/**
 * Apply mixup to named input layers.
 *
 * See:
 *
 *     Zhang, H. et al. "mixup: Beyond Empirical Risk Minimization." ICLR, 2018.
 *
 * This implementation does mixup within a single batch, per the recommendation
 * within the paper.
 *
 * This approach may create duplicate images, and so uses
 *
 *     lambda = max(lambda, 1 - lambda)
 *
 * for the mixing value.
 *
 * This recommendation comes from https://docs.fast.ai/callbacks.mixup.html
 *
 * The recommended default alpha (from the paper) is 0.4.
 */
class mixup : public callback_base {
public:
  /** Apply mixup to layers named in layers with mixup parameter alpha. */
  mixup(std::unordered_set<std::string> layers, float alpha) :
    callback_base(), m_layers(layers), m_alpha(alpha) {
    if (alpha < 0.0f) {
      LBANN_ERROR("Mixup alpha must be non-negative.");
    }
  }

  mixup* copy() const override { return new mixup(*this); }
  std::string name() const override { return "mixup"; }

  void on_forward_prop_end(model *m, Layer *l) override;

  /** @name Checkpointing */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar) {
    ar(::cereal::make_nvp(
         "BaseCallback",
         ::cereal::base_class<callback_base>(this)),
       CEREAL_NVP(m_layers),
       CEREAL_NVP(m_alpha));
  }

  ///@}

private:

  friend class cereal::access;
  mixup();

  /** Names of input layers to apply mixup to. */
  std::unordered_set<std::string> m_layers;
  /** mixup parameter. */
  float m_alpha;
};

// Builder function
std::unique_ptr<callback_base>
build_mixup_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_MIXUP_HPP
