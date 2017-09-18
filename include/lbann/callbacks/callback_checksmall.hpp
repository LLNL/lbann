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
//
// lbann_callback_checksmall .hpp .cpp - Check matrices for small values
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_CHECKSMALL_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECKSMALL_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Check matrices for whether they include any very small values to avoid
 * getting denormalized values. Denormalized values can significantly slow
 * floating point computations.
 * Since we often square values, the check is based on the square root of the
 * smallest floating point value.
 * This will kill the rank if such values are discovered.
 */
class lbann_callback_checksmall : public lbann_callback {
 public:
  using lbann_callback::on_forward_prop_end;
  using lbann_callback::on_backward_prop_end;

  lbann_callback_checksmall() : lbann_callback() {}
  lbann_callback_checksmall(const lbann_callback_checksmall&) = default;
  lbann_callback_checksmall& operator=(
    const lbann_callback_checksmall&) = default;
  lbann_callback_checksmall* copy() const {
    return new lbann_callback_checksmall(*this);
  }
  /** Check that activations are good. */
  void on_forward_prop_end(model *m, Layer *l);
  /** Check that gradients are good. */
  void on_backward_prop_end(model *m, Layer *l);
  /** Check that weights are good. */
  void on_batch_end(model *m);
  std::string name() const { return "checksmall"; }
 private:
  /** Smallest allowable value. */
  const DataType m_threshold = std::sqrt(std::numeric_limits<DataType>::min());
  /** Return true if there are no problems with m. */
  bool is_good(const AbsDistMat& m);
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_CHECKSMALL_HPP_INCLUDED
