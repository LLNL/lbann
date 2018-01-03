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
// lbann_check_reconstruction_error .hpp .cpp - Callback hooks for termination 
// after reconstruction error has reached a given value
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CHECK_RECONSTRUCTION_ERROR_HPP_INCLUDED
#define LBANN_CALLBACKS_CHECK_RECONSTRUCTION_ERROR_HPP_INCLUDED

#include <unordered_set>
#include <unordered_map>
#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Stop training after reconstruction error has reached a given value.
 */
class lbann_callback_check_reconstruction_error : public lbann_callback {
 public:
  lbann_callback_check_reconstruction_error(EvalType max_error = EvalType(1));
  lbann_callback_check_reconstruction_error(const lbann_callback_check_reconstruction_error&) = default;
  lbann_callback_check_reconstruction_error& operator=(
    const lbann_callback_check_reconstruction_error&) = default;
  lbann_callback_check_reconstruction_error* copy() const override {
    return new lbann_callback_check_reconstruction_error(*this);
  }
  /** Check if reconstruction error has reached a given value. */
  void on_epoch_end(model *m) override;
  std::string name() const override{ return "check reconstruction error"; }
 private:
  /** maximum error value, default is 1.0 */
  EvalType m_max_error;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CHECK_RECONSTRUCTION_ERROR_HPP_INCLUDED
