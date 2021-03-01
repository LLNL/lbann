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
// debug .hpp .cpp - Callback hooks to debug LBANN
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DEBUG_IO_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DEBUG_IO_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/layers/io/input_layer.hpp"
#include <chrono>
#include <vector>

namespace lbann {
namespace callback {

/**
 * Print status updates on where training is.
 */
class debug_io : public callback_base {
 public:
  using callback_base::on_forward_prop_begin;
  using callback_base::on_forward_prop_end;
  using callback_base::on_backward_prop_begin;
  using callback_base::on_backward_prop_end;
  using callback_base::on_evaluate_forward_prop_begin;
  using callback_base::on_evaluate_forward_prop_end;

  /**
   * Debug a particular phase; use invalid to debug every phase.
   */
  debug_io(execution_mode phase = execution_mode::invalid,
                          int debug_lvl = 0) :
    callback_base(1),
    m_debug_phase(phase),
    m_debug_lvl(debug_lvl) {}
  debug_io(const debug_io&) = default;
  debug_io& operator=(
    const debug_io&) = default;
  debug_io* copy() const override { return new debug_io(*this); }
  /** Print that a training epoch is being started. */
  void on_epoch_begin(model *m) override;
  /** Print that forward prop for a layer is beginning. */
  void on_forward_prop_begin(model *m, Layer *l) override;

  /** Print I/O details at the beginning of validation. */
  void on_validation_begin(model *m) override;
  /** Print that an evaluation forward prop is beginning. */
  void on_evaluate_forward_prop_begin(model *m, Layer *l) override;

  /** Print I/O details at the beginning of testing. */
  void on_test_begin(model *m) override;

  /** Common format for printing I/O stats at the start of a mini-batch */
  void print_fp_start(model *m, input_layer<DataType> *input);
  /** Common format for printing I/O stats at the start of a phase */
  void print_phase_start(model *m, execution_mode mode);

  std::string name() const override { return "debug_io"; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar);

  ///@}

 private:
  /** The phase to debug. */
  execution_mode m_debug_phase;
  int m_debug_lvl; /** Debugging level: 0 - epoch begin, 1 - fwd prop */
};

// Builder function
std::unique_ptr<callback_base>
build_debug_io_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DEBUG_IO_HPP_INCLUDED
