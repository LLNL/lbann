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
//
// check_small .hpp .cpp - Check matrices for small values
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_CHECK_SMALL_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECK_SMALL_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 * Check matrices for whether they include any very small values to avoid
 * getting denormalized values. Denormalized values can significantly slow
 * floating point computations.
 * Since we often square values, the check is based on the square root of the
 * smallest floating point value.
 * This will kill the rank if such values are discovered.
 */
class check_small : public callback_base
{
public:
  using callback_base::on_backward_prop_end;
  using callback_base::on_forward_prop_end;

  check_small() = default;
  check_small(const check_small&) = default;
  check_small& operator=(const check_small&) = default;
  check_small* copy() const override { return new check_small(*this); }
  /** Check that activations are good. */
  void on_forward_prop_end(model* m, Layer* l) override;
  /** Check that gradients are good. */
  void on_backward_prop_end(model* m) override;
  /** Check that weights are good. */
  void on_batch_end(model* m) override;
  std::string name() const override { return "check_small"; }

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
LBANN_ADD_DEFAULT_CALLBACK_BUILDER(check_small,
                                   build_check_small_callback_from_pbuf)

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_CHECK_SMALL_HPP_INCLUDED
