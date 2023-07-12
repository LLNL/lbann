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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_PROFILER_CALIPER_HPP_INCLUDED
#define LBANN_CALLBACKS_PROFILER_CALIPER_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"

#ifdef LBANN_HAS_CALIPER
#include <caliper/cali-manager.h>
#endif // LBANN_HAS_CALIPER

namespace lbann {
namespace callback {

#ifdef LBANN_HAS_CALIPER

class profiler_caliper : public callback_base
{
public:
  profiler_caliper(bool skip_init = true);
  profiler_caliper(const profiler_caliper&) = default;
  ~profiler_caliper();

  profiler_caliper* copy() const override
  {
    return new profiler_caliper(*this);
  }

  void on_epoch_begin(model* m) override;
  void on_epoch_end(model* m) override;
  void on_validation_begin(model* m) override;
  void on_validation_end(model* m) override;
  void on_test_begin(model* m) override;
  void on_test_end(model* m) override;
  void on_batch_begin(model* m) override;
  void on_batch_end(model* m) override;
  void on_batch_evaluate_begin(model* m) override;
  void on_batch_evaluate_end(model* m) override;
  void on_forward_prop_begin(model* m) override;
  void on_forward_prop_end(model* m) override;
  void on_evaluate_forward_prop_begin(model* m) override;
  void on_evaluate_forward_prop_end(model* m) override;
  void on_backward_prop_begin(model* m) override;
  void on_backward_prop_end(model* m) override;
  void on_forward_prop_begin(model* m, Layer* l) override;
  void on_forward_prop_end(model* m, Layer* l) override;
  void on_evaluate_forward_prop_begin(model* m, Layer* l) override;
  void on_evaluate_forward_prop_end(model* m, Layer* l) override;
  void on_backward_prop_begin(model* m, Layer* l) override;
  void on_backward_prop_end(model* m, Layer* l) override;
  void on_optimize_begin(model* m) override;
  void on_optimize_end(model* m) override;
  void on_optimize_begin(model* m, weights* w) override;
  void on_optimize_end(model* m, weights* w) override;
  std::string name() const override { return "profiler_caliper"; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

private:
  void start_profiling();

  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  struct manager_wrapper
  {
    manager_wrapper();
    ~manager_wrapper();
    void start();
    cali::ConfigManager manager;
    bool started = false;
  }; // struct manager_wrapper
  manager_wrapper m_manager;

  /** Whether to skip initial iterations. */
  bool m_skip_init; // default is to skip first epoch
};

#endif // ifdef LBANN_HAS_CALIPER

// Builder function
std::unique_ptr<callback_base> build_profiler_caliper_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_PROFILER_CALIPER_HPP_INCLUDED
