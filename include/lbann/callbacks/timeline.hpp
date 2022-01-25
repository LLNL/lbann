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
// callback_timeline .hpp .cpp - Callback hooks to record a timeline of runtime
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_TIMELINE_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_TIMELINE_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

#include <unordered_map>
#include <vector>

namespace lbann {
namespace callback {

/**
 * Record a timeline of training runtime on each rank and output it to a
 * logfile for external processing.
 * The logfile is named timeline.m\<model-rank\>.\<rank\>.txt.
 * Each line is a separate event, written as name:start-time:end-time.
 * Times are relative to the beginning of training.
 */
class timeline : public callback_base {
 public:
  timeline(std::string outdir) : callback_base(1),
                                                m_outdir(outdir) {}
  timeline(const timeline&) = default;
  timeline& operator=(const timeline&) = default;
  timeline* copy() const override {
    return new timeline(*this);
  }
  std::string name() const override { return "timeline"; }
  void on_train_begin(model *m) override;
  void on_train_end(model *m) override;

  using callback_base::on_forward_prop_begin;
  using callback_base::on_forward_prop_end;
  using callback_base::on_backward_prop_begin;
  using callback_base::on_backward_prop_end;
  using callback_base::on_optimize_begin;
  using callback_base::on_optimize_end;

  void on_forward_prop_begin(model *m, Layer *l) override;
  void on_forward_prop_end(model *m, Layer *l) override;
  void on_backward_prop_begin(model *m, Layer *l) override;
  void on_backward_prop_end(model *m, Layer *l) override;
  void on_optimize_begin(model *m, weights *w) override;
  void on_optimize_end(model *m, weights *w) override;

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar);

  ///@}

 private:

  friend class cereal::access;
  timeline();

  /// Get time relative to the start time.
  EvalType get_rel_time() const { return get_time() - m_start_time; }

  /// Directory to write output to.
  std::string m_outdir;
  /// Time training started; all times are relative to this.
  EvalType m_start_time = EvalType(0);
  /// Time the current layer's forward pass started.
  EvalType m_fp_start_time = EvalType(0);
  /// Time the current layer's backward pass started.
  EvalType m_bp_start_time = EvalType(0);
  /// Time the current weights' optimization pass started.
  EvalType m_opt_start_time = EvalType(0);
  /// Store (relative) timing information.
  std::unordered_map<std::string, std::vector<std::pair<EvalType, EvalType>>> m_fp_times;
  std::unordered_map<std::string, std::vector<std::pair<EvalType, EvalType>>> m_bp_times;
  std::unordered_map<std::string, std::vector<std::pair<EvalType, EvalType>>> m_opt_times;
};

// Builder function
std::unique_ptr<callback_base>
build_timeline_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_TIMELINE_HPP_INCLUDED
