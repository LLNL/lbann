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

#ifndef LBANN_CALLBACKS_CALLBACK_SET_WEIGHTS_VALUE_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SET_WEIGHTS_VALUE_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/** @brief Set values in a weights object at a given training step
 *
 *  @todo Support weights with arbitrary data types. Currently only
 *  floats are supported.
 */
class set_weights_value : public callback_base
{
public:
  /**
   *  @param weights_name Name of weights object
   *  @param value Value to set weights
   *  @param step Mini-batch step at which to set weights value
   */
  set_weights_value(std::string weights_name, double value, size_t step);
  set_weights_value(const set_weights_value&) = default;
  set_weights_value& operator=(const set_weights_value&) = default;

  set_weights_value* copy() const override;
  std::string name() const override;

  void on_batch_begin(model* m) override;

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
  set_weights_value();

  /** @brief Name of weights object. */
  std::string m_weights_name;
  /** @brief Value to set weights. */
  double m_value;
  /** @brief Mini-batch step at which to set weights value. */
  size_t m_step;
};

// Builder function
std::unique_ptr<callback_base> build_set_weights_value_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_SET_WEIGHTS_VALUE_HPP_INCLUDED
