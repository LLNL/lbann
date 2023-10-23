////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
// gradient_clipping .hpp .cpp - Callbacks to clip gradient values in training
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_GRADIENT_CLIPPING_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_GRADIENT_CLIPPING_HPP_INCLUDED

#include <unordered_set>
#include <utility>

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 * @brief Clip gradients whose norm is larger than a user-defined value by
 * dividing them.
 */
class clip_gradient_norm : public callback_base
{
public:
  using callback_base::on_backward_prop_end;

  /**
   * @param weights Parameters whose gradient to clip, or empty for all
   * @param global_norm Whether to clip according to the norm of all parameters
   *                    or each one separately
   * @param value Value to clip to
   */
  clip_gradient_norm(std::vector<std::string> weights,
                     bool global_norm = false,
                     float value = 1.0f)
    : callback_base(1),
      m_weight_names(std::move(weights)),
      m_global_norm(global_norm),
      m_value(value)
  {}
  clip_gradient_norm(const clip_gradient_norm&) = default;
  clip_gradient_norm& operator=(const clip_gradient_norm&) = default;
  void setup(model* m) override;
  clip_gradient_norm* copy() const override
  {
    return new clip_gradient_norm(*this);
  }
  void on_backward_prop_end(model* m) override;
  std::string name() const override { return "clip gradient norm"; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  friend class cereal::access;
  clip_gradient_norm();

  /** @brief Parameter names whose gradients to clip. */
  std::vector<std::string> m_weight_names;

  /** @brief Whether to clip according to the norm of all parameters. */
  bool m_global_norm;

  /** @brief Value to clip to. */
  float m_value;

  /** Weights to update. */
  std::unordered_set<weights*> m_weights;
};

// Builder function
std::unique_ptr<callback_base> build_clip_gradient_norm_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_GRADIENT_CLIPPING_HPP_INCLUDED
