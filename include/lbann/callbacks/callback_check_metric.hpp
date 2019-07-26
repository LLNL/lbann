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

#ifndef LBANN_CALLBACKS_CALLBACK_CHECK_METRIC_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECK_METRIC_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <set>

namespace lbann {

/** Metric checking callback.
 *  Checks if a metric value falls within an expected range.
 */
class lbann_callback_check_metric : public lbann_callback {
public:

  lbann_callback_check_metric(std::string metric_name,
                              std::set<execution_mode> modes,
                              EvalType lower_bound,
                              EvalType upper_bound,
                              bool error_on_failure);
  lbann_callback_check_metric* copy() const override {
    return new lbann_callback_check_metric(*this);
  }
  std::string name() const override { return "check metric"; }

  void on_epoch_end(model* m) override      { check_metric(*m); }
  void on_validation_end(model* m) override { check_metric(*m); }
  void on_test_end(model* m) override       { check_metric(*m); }

private:

  /** Metric name. */
  std::string m_metric_name;

  /** Execution modes with metric checks. */
  std::set<execution_mode> m_modes;

  /** Lower bound for metric value. */
  EvalType m_lower_bound;
  /** Upper bound for metric value. */
  EvalType m_upper_bound;

  /** Whether to throw an exception if metric check fails. */
  bool m_error_on_failure;

  /** Perform metric check.
   *  Does nothing if current execution mode is not in m_modes;
   */
  void check_metric(const model& m) const;

};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_check_metric_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_CHECK_METRIC_HPP_INCLUDED
