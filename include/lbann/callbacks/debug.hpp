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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DEBUG_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DEBUG_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

#include <set>

namespace lbann {
namespace callback {

/**
 * @brief Phase specific "printf debugging"
 *
 * Print verbose status updates to standard error stream.
 * This callback is useful for "printf debugging."
 *
 * Takes a prototext parameter @c phase: train | validate | test | \<empty\>
 * if \<empty\> will print messages for all phases
 *
 */
class debug : public callback_base {
 public:

  /** @brief Constructor.
   *
   *  If modes is empty, status updates will be printed for all
   *  execution modes.
   */
  debug(std::set<execution_mode> modes={}) :
    m_modes(std::move(modes)) {}
  debug(const debug&) = default;
  debug& operator=(const debug&) = default;
  debug* copy() const override { return new debug(*this); }
  std::string name() const override { return "debug"; }

  /** @brief Print that a batch is beginning. */
  void on_batch_begin(model *m) override;
  /** @brief Print that a batch is ending. */
  void on_batch_end(model *m) override;
  /** @brief Print that a layer's forward prop is beginning. */
  void on_batch_evaluate_begin(model *m) override;
  /** @brief Print that a layer's forward prop is ending. */
  void on_batch_evaluate_end(model *m) override;

  using callback_base::on_forward_prop_begin;
  using callback_base::on_forward_prop_end;
  using callback_base::on_backward_prop_begin;
  using callback_base::on_backward_prop_end;
  using callback_base::on_evaluate_forward_prop_begin;
  using callback_base::on_evaluate_forward_prop_end;

  /** @brief Print that a layer's forward prop is beginning. */
  void on_forward_prop_begin(model *m, Layer *l) override;
  /** @brief Print that a layer's forward prop is ending. */
  void on_forward_prop_end(model *m, Layer *l) override;
  /** @brief Print that a layer's backward prop is beginning. */
  void on_backward_prop_begin(model *m, Layer *l) override;
  /** @brief Print that a layer's backward prop is ending. */
  void on_backward_prop_end(model *m, Layer *l) override;
  /** @brief Print that a layer's backward prop is beginning. */
  void on_evaluate_forward_prop_begin(model *m, Layer *l) override;
  /** @brief Print that a layer's backward prop is ending. */
  void on_evaluate_forward_prop_end(model *m, Layer *l) override;

  /** @brief Print that a weights' optimization step is beginning. */
  void on_optimize_begin(model *m, weights *w) override;
  /** @brief Print that a weights' optimization step is ending. */
  void on_optimize_end(model *m, weights *w) override;

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar);

  ///@}

 private:

  /** @brief Execution modes for which status updates will be printed.
   *
   *  If empty, status updates are printed for all execution modes.
   */
  std::set<execution_mode> m_modes;

};

// Builder function
std::unique_ptr<callback_base>
build_debug_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_DEBUG_HPP_INCLUDED
