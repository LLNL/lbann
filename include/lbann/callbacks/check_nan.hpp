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
// check_nan .hpp .cpp - Check matrices for invalid numbers
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_CHECK_NAN_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECK_NAN_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 * Check matrices for whether they include any NaNs or infs to help debugging.
 * This will kill the rank if such values are discovered.
 */
class check_nan : public callback_base
{
public:
  using callback_base::on_backward_prop_end;
  using callback_base::on_forward_prop_end;

  check_nan() = default;
  check_nan(const check_nan&) = default;
  check_nan& operator=(const check_nan&) = default;
  check_nan* copy() const override { return new check_nan(*this); }
  /** Check that activations are good. */
  void on_forward_prop_end(model* m, Layer* l) override;
  /** Check that error signals are good. */
  void on_backward_prop_end(model* m, Layer* l) override;
  /** Check that gradients are good. */
  void on_backward_prop_end(model* m) override;
  /** Check that weights are good. */
  void on_batch_end(model* m) override;
  std::string name() const override { return "check_nan"; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;
};

// Builder function
LBANN_ADD_DEFAULT_CALLBACK_BUILDER(check_nan,
                                   build_check_nan_callback_from_pbuf)

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_CHECK_NAN_HPP_INCLUDED
