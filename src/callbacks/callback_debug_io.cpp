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
// lbann_callback_debug .hpp .cpp - Callback hooks to debug LBANN
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_debug_io.hpp"

void lbann::lbann_callback_debug_io::on_batch_begin(model *m) {
  if(m_debug_phase == execution_mode::invalid || m_debug_phase == m->get_execution_mode()) {
    std::cout << "Phase: " << _to_string(m->get_execution_mode()) << " starting batch" << std::endl;
  }
}

void lbann::lbann_callback_debug_io::on_forward_prop_begin(model *m, Layer *l) {
  if (!dynamic_cast<input_layer*>(l) || l->get_index() != 0) {
    return;
  }

  input_layer *input = dynamic_cast<input_layer*>(l);

  if(m_debug_phase == execution_mode::invalid || m_debug_phase == m->get_execution_mode()) {
    std::cout << "[" << m->get_comm()->get_model_rank() << "." << m->get_comm()->get_rank_in_model() << "] @" << m->get_cur_epoch() << "." << m->get_cur_step() << " Phase: " << _to_string(m->get_execution_mode()) << " starting forward propagation for layer " << l->get_index() << " name: " << l->get_name() 
              << " iteration: " << input->get_data_reader()->get_current_mini_batch_index()
              << " of " << input->get_num_iterations_per_epoch()
              << " bs=" << input->get_current_mini_batch_size() << "/"
              << input->get_current_global_mini_batch_size() 
              << " @" << input->get_data_reader()->get_position()
      //              << " %" << input->get_data_reader()->get_batch_stride()
              << " ^" << input->get_data_reader()->get_sample_stride()
              << std::endl;
  }

  /// BVE Note - what is hte role of hte current mini-batch index
  /// versus the current position
  /// I think that the reset mini batch index may be off
}

#if 0

////////////////////////////////////////////////////////////////////////////////
// Evaluation phase debugging
////////////////////////////////////////////////////////////////////////////////
void lbann::lbann_callback_debug_io::on_batch_evaluate_begin(model *m) {
  if(m_debug_phase == execution_mode::invalid || m_debug_phase == m->get_execution_mode()) {
    int64_t step;
    switch(m->get_execution_mode()) {
    case execution_mode::validation:
      step = m->get_cur_validation_step();
      break;
    case execution_mode::testing:
      step = m->get_cur_testing_step();
      break;
    default:
      throw lbann_exception("Illegal execution mode in evaluate forward prop function");
    }
    std::cout << "[" << m->get_comm()->get_model_rank() << "." << m->get_comm()->get_rank_in_model() << "] @" << 0 << "." << step << " Phase: " << _to_string(m->get_execution_mode()) << " starting batch" << std::endl;
  }
}

void lbann::lbann_callback_debug_io::on_batch_evaluate_end(model *m) {
  if(m_debug_phase == execution_mode::invalid || m_debug_phase == m->get_execution_mode()) {
    int64_t step;
    switch(m->get_execution_mode()) {
    case execution_mode::validation:
      step = m->get_cur_validation_step();
      break;
    case execution_mode::testing:
      step = m->get_cur_testing_step();
      break;
    default:
      throw lbann_exception("Illegal execution mode in evaluate forward prop function");
    }
    std::cout << "[" << m->get_comm()->get_model_rank() << "." << m->get_comm()->get_rank_in_model() << "] @" << 0 << "." << step << " Phase: " << _to_string(m->get_execution_mode()) << " ending batch" << std::endl;
  }
}

void lbann::lbann_callback_debug_io::on_evaluate_forward_prop_begin(model *m, Layer *l) {
  if(m_debug_phase == execution_mode::invalid || m_debug_phase == m->get_execution_mode()) {
    int64_t step;
    switch(m->get_execution_mode()) {
    case execution_mode::validation:
      step = m->get_cur_validation_step();
      break;
    case execution_mode::testing:
      step = m->get_cur_testing_step();
      break;
    default:
      throw lbann_exception("Illegal execution mode in evaluate forward prop function");
    }
    std::cout << "[" << m->get_comm()->get_model_rank() << "." << m->get_comm()->get_rank_in_model() << "] @" << 0 << "." << step << " Phase: " << _to_string(m->get_execution_mode()) << " starting forward propagation for layer " << l->get_index() << " name: " << l->get_name() << std::endl;
  }
}

void lbann::lbann_callback_debug_io::on_evaluate_forward_prop_end(model *m, Layer *l) {
  if(m_debug_phase == execution_mode::invalid || m_debug_phase == m->get_execution_mode()) {
    int64_t step;
    switch(m->get_execution_mode()) {
    case execution_mode::validation:
      step = m->get_cur_validation_step();
      break;
    case execution_mode::testing:
      step = m->get_cur_testing_step();
      break;
    default:
      throw lbann_exception("Illegal execution mode in evaluate forward prop function");
    }
    std::cout << "[" << m->get_comm()->get_model_rank() << "." << m->get_comm()->get_rank_in_model() << "] @" << 0 << "." << step << " Phase: " << _to_string(m->get_execution_mode()) << "   ending forward propagation for layer " << l->get_index() << " name: " << l->get_name() << std::endl;
  }
}
#endif
