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
// lbann_callback_print .hpp .cpp - Callback hooks to print information
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_PRINT_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_PRINT_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/** Periodically print computational results.
 *  Prints average objective function value and metric scores after
 *  each training epoch and evaluation.
 */
class lbann_callback_print : public lbann_callback {
 public:
  lbann_callback_print(int batch_interval = 1, bool print_global_stat_only=false) :
  lbann_callback(batch_interval), m_print_global_stat_only(print_global_stat_only) {}
  lbann_callback_print(const lbann_callback_print&) = default;
  lbann_callback_print& operator=(const lbann_callback_print&) = default;
  lbann_callback_print* copy() const override { return new lbann_callback_print(*this); }
  void setup(model *m) override;
  void on_epoch_begin(model *m) override;
  void on_epoch_end(model *m) override;
  void on_validation_end(model *m) override;
  void on_test_end(model *m) override;
  std::string name() const override { return "print"; }

 private:
  /** Print objective function and metrics to standard output. */
  void report_results(model *m);
  bool m_print_global_stat_only;

};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_PRINT_HPP_INCLUDED
