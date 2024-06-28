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
// memory_profiler .hpp .cpp - Itemized memory usage profiling.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_MEMORY_PROFILER_HPP_INCLUDED
#define LBANN_CALLBACKS_MEMORY_PROFILER_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

#include <map>

namespace lbann {
namespace callback {

/**
 * @brief Returns the currently used memory, or 0 if LBANN was not compiled with
 * GPU support.
 */
size_t get_used_gpu_memory();

/**
 * Memory usage profiling
 */
class memory_profiler : public callback_base
{
public:
  memory_profiler(bool detailed_first_step = false);
  memory_profiler(const memory_profiler&) = default;
  memory_profiler& operator=(const memory_profiler&) = default;
  ~memory_profiler();
  memory_profiler* copy() const override { return new memory_profiler(*this); }

  // Used for coarse-grained accounting
  void on_setup_begin(model* m) override;
  void on_setup_end(model* m) override;
  void on_setup_begin(model* m, Layer* l) override;
  void on_setup_end(model* m, Layer* l) override;
  void on_batch_end(model* m) override;

  // Used for detailed first-step accounting
  void on_forward_prop_begin(model* m) override;
  void on_forward_prop_end(model* m) override;
  void on_backward_prop_begin(model* m) override;
  void on_backward_prop_end(model* m) override;
  void on_optimize_begin(model* m) override;
  void on_optimize_end(model* m) override;

  // Used for layer-wise accounting in the first few steps
  void on_forward_prop_begin(model* m, Layer* l) override;
  void on_forward_prop_end(model* m, Layer* l) override;
  void on_backward_prop_begin(model* m, Layer* l) override;
  void on_backward_prop_end(model* m, Layer* l) override;

  std::string name() const override { return "memory profiler"; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

private:
  /** Prints the memory usage layer breakdown of a model. */
  void report_mem_usage(model* m);

  /** Performs first step detailed memory usage accounting. */
  void first_step_accounting(model* m, const std::string& msg);

  /** Performs peak memory usage collection in third step. */
  void collect_peak_usage();

  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /** Whether to print layer-wise accounting if more allocations were
   * detected in the first step.
   */
  bool m_detailed_first_step;

  /** Initial memory usage in bytes */
  size_t m_initial_memory_usage;

  /** Unaccounted memory in bytes during layer setup */
  std::map<Layer*, size_t> m_unaccounted_setup_layer;

  /** Unaccounted memory in bytes during forward propagation */
  std::map<Layer*, size_t> m_unaccounted_fp_layer;

  /** Unaccounted memory in bytes during backpropagation */
  std::map<Layer*, size_t> m_unaccounted_bp_layer;

  /** Activation sizes in bytes per layer */
  std::map<Layer*, size_t> m_act_sizes;

  /** Activation shape report per layer */
  std::map<Layer*, std::string> m_act_report;

  /** Current step, used for tracking memory usage. */
  int m_current_step;

  /** Tracking of raw memory usage across the first three steps to identify
   * leaks.
   */
  size_t m_setup_end_usage, m_step0_usage, m_step1_usage, m_step2_usage,
    m_peak_mem_usage;
};

// Builder function
std::unique_ptr<callback_base>
build_memory_profiler_callback_from_pbuf(const google::protobuf::Message&,
                                         std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_MEMORY_PROFILER_HPP_INCLUDED
