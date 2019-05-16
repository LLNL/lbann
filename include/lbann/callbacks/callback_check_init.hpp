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
// lbann_callback_check_init .hpp .cpp - Check multi-model init
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_CHECK_INIT_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECK_INIT_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Verify that every model uses the same initialization.
 */
class lbann_callback_check_init : public lbann_callback {
 public:
  lbann_callback_check_init() : lbann_callback() {}
  lbann_callback_check_init(const lbann_callback_check_init&) = default;
  lbann_callback_check_init& operator=(
    const lbann_callback_check_init&) = default;
  lbann_callback_check_init* copy() const override {
    return new lbann_callback_check_init(*this);
  }
  /** Check initializations. */
  void on_train_begin(model *m) override;
  std::string name() const override { return "check init"; }
 private:
  /** Return true if x == y. */
  bool check_equal(const AbsMat& x, const AbsMat& y) const;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_CHECK_INIT_HPP_INCLUDED
