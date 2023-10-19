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
// debug .hpp .cpp - Callback hooks to debug LBANN
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/debug_io.hpp"

#include "lbann/base.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/serialize.hpp"

#include "lbann/proto/callbacks.pb.h"
#include <iostream>
#include <memory>

namespace lbann {
namespace callback {

template <class Archive>
void debug_io::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_debug_phase),
     CEREAL_NVP(m_debug_lvl));
}

void debug_io::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_debug_io();
  msg->set_phase(to_string(m_debug_phase));
  msg->set_lvl(m_debug_lvl);
}

/// BVE FIXME @todo The use of execution_mode invalid needs to be reconsidered
void debug_io::on_epoch_begin(model* m)
{
  if (m_debug_phase == execution_mode::invalid ||
      m_debug_phase == execution_mode::training) {
    print_phase_start(m, execution_mode::training);
  }
}

void debug_io::on_forward_prop_begin(model* m, Layer* l)
{
  auto* input = dynamic_cast<input_layer<DataType>*>(l);
  if (input == nullptr || m_debug_lvl < 1) {
    return;
  }

  const auto& c = m->get_execution_context();
  auto mode = c.get_execution_mode();
  if (m_debug_phase == execution_mode::invalid || m_debug_phase == mode) {
    print_fp_start(m, input);
  }
  /// BVE Note - what is hte role of hte current mini-batch index
  /// versus the current position
  /// I think that the reset mini batch index may be off
}

void debug_io::print_fp_start(model* m, input_layer<DataType>* input)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  const data_coordinator& dc = get_const_trainer().get_data_coordinator();
  const auto& step = c.get_step();
  const auto mode = c.get_execution_mode();
  std::cout
    << "[" << m->get_comm()->get_trainer_rank() << "."
    << m->get_comm()->get_rank_in_trainer() << "] @" << c.get_epoch() << "."
    << step << " Phase: " << to_string(mode)
    << " starting forward propagation for layer " << input->get_name()
    << " type: " << input->get_type() << " iteration: "
    << dc.get_data_reader(mode)->get_current_mini_batch_index() << " of "
    << dc.get_num_iterations_per_epoch(mode) << " loading idx "
    << dc.get_data_reader(mode)->get_loaded_mini_batch_index()
    << " bs=" << dc.get_current_mini_batch_size(mode) << " @"
    << dc.get_data_reader(mode)->get_position()
    //              << " %" << input->get_data_reader()->get_batch_stride()
    << " ^" << dc.get_data_reader(mode)->get_sample_stride() << std::endl;
}

//  179i @ 300s (=5m*60s) + 1i @ 100s (=5m*45s):offset <- num models
void debug_io::print_phase_start(model* m, execution_mode mode)
{
  const auto& c = m->get_execution_context();
  const data_coordinator& dc = get_const_trainer().get_data_coordinator();
  // Get data reader from first input layer in model
  generic_data_reader* data_reader = dc.get_data_reader(mode);
  const auto& step = c.get_step();

  std::cout << "[" << m->get_comm()->get_trainer_rank() << "."
            << m->get_comm()->get_rank_in_trainer() << "] @" << 0 << "." << step
            << " Starting Phase: " << to_string(mode) << " "
            << (data_reader->get_num_iterations_per_epoch() - 1) << "i @ "
            << data_reader->get_mini_batch_size() << "s [+"
            << data_reader->get_stride_to_next_mini_batch() << "s]) + 1i @ "
            << data_reader->get_last_mini_batch_size() << "s [+"
            << data_reader->get_stride_to_last_mini_batch() << "s]):"
            << " base offset " << data_reader->get_base_offset() << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// Evaluation phase debugging
////////////////////////////////////////////////////////////////////////////////
void debug_io::on_validation_begin(model* m)
{
  if (m_debug_phase == execution_mode::invalid ||
      m_debug_phase == execution_mode::validation) {
    print_phase_start(m, execution_mode::validation);
  }
}

void debug_io::on_evaluate_forward_prop_begin(model* m, Layer* l)
{
  auto* input = dynamic_cast<input_layer<DataType>*>(l);
  if (input == nullptr || m_debug_lvl < 1) {
    return;
  }

  const auto& c = m->get_execution_context();
  auto mode = c.get_execution_mode();
  if (m_debug_phase == execution_mode::invalid || m_debug_phase == mode) {
    print_fp_start(m, input);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Testing phase debugging
////////////////////////////////////////////////////////////////////////////////
void debug_io::on_test_begin(model* m)
{
  if (m_debug_phase == execution_mode::invalid ||
      m_debug_phase == execution_mode::testing) {
    print_phase_start(m, execution_mode::testing);
  }
}

std::unique_ptr<callback_base>
build_debug_io_callback_from_pbuf(const google::protobuf::Message& proto_msg,
                                  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDebugIO&>(proto_msg);
  const auto& phase = exec_mode_from_string(params.phase());
  const auto& lvl = params.lvl();
  switch (phase) {
  case execution_mode::training:
  case execution_mode::validation:
  case execution_mode::testing:
    return std::make_unique<debug_io>(phase, lvl);
  default:
    return std::make_unique<debug_io>();
  }
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::debug_io
#define LBANN_CLASS_LIBNAME callback_debug_io
#include <lbann/macros/register_class_with_cereal.hpp>
