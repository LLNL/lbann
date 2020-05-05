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
// set_weights_value .hpp .cpp - Callbacks to set weights value
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_SET_WEIGHTS_VALUE_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SET_WEIGHTS_VALUE_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 * Set weights value.
 */
class set_weights_value : public callback_base {
 public:
  /**
   * @param weight_name The name of weight to be set.
   * @param weight_value The new value for weight_name.
   * @param epoch_interval The epoch at which to set new value.
   *
   */
  set_weights_value(std::string weight_name, float weight_value,El::Int epoch_interval=0) :
    callback_base(), m_weight_name(std::move(weight_name)),
    m_weight_value(weight_value),
    m_epoch_interval(std::max(El::Int(0),epoch_interval)) {}
  set_weights_value(const set_weights_value&) = default;
  set_weights_value& operator=(
    const set_weights_value&) = default;
  set_weights_value* copy() const override {
    return new set_weights_value(*this);
  }
  void on_epoch_begin(model *m) override;
  std::string name() const override { return "set weights value"; }
 private:
  /** weight name. */
  std::string m_weight_name;
  /** new weight value */
  float m_weight_value;
  /** Epoch interval at which to set weight value*/
  El::Int m_epoch_interval;
};

// Builder function
std::unique_ptr<callback_base>
build_set_weights_value_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_SET_WEIGHTS_VALUE_HPP_INCLUDED
