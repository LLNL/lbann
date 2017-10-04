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

/// BVE FIXME @todo The use of execution_mode invalid needs to be reconsidered
void lbann::lbann_callback_debug_io::on_epoch_begin(model *m) {
  if(m_debug_phase == execution_mode::invalid || m_debug_phase == execution_mode::training) {
    print_phase_start(m, execution_mode::training);
  }
}

void lbann::lbann_callback_debug_io::on_forward_prop_begin(model *m, Layer *l) {
  if (!dynamic_cast<input_layer*>(l) || l->get_index() != 0 || m_debug_lvl < 1) {
    return;
  }

  input_layer *input = dynamic_cast<input_layer*>(l);

  if(input->current_root_rank() == 0) {
    if(m->get_comm()->get_rank_in_model() < input->get_data_reader()->get_num_parallel_readers() && !input->is_local_reader_done()) {
      if(m_debug_phase == execution_mode::invalid || m_debug_phase == m->get_execution_mode()) {
        print_fp_start(m, input);
      }
    }
  }
  /// BVE Note - what is hte role of hte current mini-batch index
  /// versus the current position
  /// I think that the reset mini batch index may be off
}

void lbann::lbann_callback_debug_io::print_fp_start(model *m, input_layer *input) {
  int64_t step;
  switch(m->get_execution_mode()) {
  case execution_mode::training:
    step = m->get_cur_step();
    break;
  case execution_mode::validation:
    step = m->get_cur_validation_step();
    break;
  case execution_mode::testing:
    step = m->get_cur_testing_step();
    break;
  default:
    throw lbann_exception("Illegal execution mode in evaluate forward prop function");
  }
  std::cout << "[" << m->get_comm()->get_model_rank() 
            << "." << m->get_comm()->get_rank_in_model() 
            << "] @" << m->get_cur_epoch() << "." << step 
            << " Phase: " << _to_string(m->get_execution_mode()) 
            << " starting forward propagation for layer " << input->get_index() 
            << " name: " << input->get_name() 
            << " iteration: " << input->get_data_reader()->get_current_mini_batch_index()
            << " of " << input->get_num_iterations_per_epoch()
            << " loading idx " << input->get_data_reader()->get_loaded_mini_batch_index()
            << " bs=" << input->get_current_mini_batch_size() << "/"
            << input->get_current_global_mini_batch_size() 
            << " @" << input->get_data_reader()->get_position()
    //              << " %" << input->get_data_reader()->get_batch_stride()
            << " ^" << input->get_data_reader()->get_sample_stride()
            << " root=" << input->current_root_rank()
            << std::endl;
}

//  179i @ 300s (=5m*60s) + 1i @ 100s (=5m*45s):offset <- num models
void lbann::lbann_callback_debug_io::print_phase_start(model *m, execution_mode mode) {
  std::vector<Layer *>layers = m->get_layers();
  input_layer *input = dynamic_cast<input_layer*>(layers[0]);
  generic_data_reader *data_reader=input->get_data_reader(mode);

  int64_t step;
  switch(mode) {
  case execution_mode::training:
    step = m->get_cur_step();
    break;
  case execution_mode::validation:
    step = m->get_cur_validation_step();
    break;
  case execution_mode::testing:
    step = m->get_cur_testing_step();
    break;
  default:
    throw lbann_exception("Illegal execution mode in evaluate forward prop function");
  }

  if(data_reader->get_rank() < data_reader->get_num_parallel_readers()) {
    std::cout << "[" << m->get_comm()->get_model_rank() 
              << "." << m->get_comm()->get_rank_in_model() 
              << "] @" << 0 << "." << step 
              << " Starting Phase: " << _to_string(mode) 
              << " " << (data_reader->get_num_iterations_per_epoch() - 1)
              << "i @ " << data_reader->get_global_mini_batch_size()
              << "s (=" << m->get_comm()->get_num_models()
              << "m *" << data_reader->get_mini_batch_size()
              << "s [+" << data_reader->get_stride_to_next_mini_batch()
              << "s]) + 1i @ " << data_reader->get_global_last_mini_batch_size()
              << "s (=" << m->get_comm()->get_num_models()
              << "m *" << data_reader->get_last_mini_batch_size()
              << "s [+" << data_reader->get_stride_to_last_mini_batch()
              << "s]):" 
              <<" base offset "<< data_reader->get_base_offset() 
              << " model offset " << data_reader->get_model_offset() 
              << " par. readers = " << data_reader->get_num_parallel_readers()
              << "r"
              << std::endl;
  }else {
    std::cout << "[" << m->get_comm()->get_model_rank() 
              << "." << m->get_comm()->get_rank_in_model() 
              << "] @" << 0 << "." << step 
              << " Starting Phase: " << _to_string(mode) 
              << " " << (data_reader->get_num_iterations_per_epoch())
              << "i "
              << " par. readers = " << data_reader->get_num_parallel_readers()
              << "r (Inactive Reader)"
              << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Evaluation phase debugging
////////////////////////////////////////////////////////////////////////////////
void lbann::lbann_callback_debug_io::on_validation_begin(model *m) {
  if(m_debug_phase == execution_mode::invalid || m_debug_phase == execution_mode::validation) {
    print_phase_start(m, execution_mode::validation);
  }
}

void lbann::lbann_callback_debug_io::on_evaluate_forward_prop_begin(model *m, Layer *l) {
  if (!dynamic_cast<input_layer*>(l) || l->get_index() != 0 || m_debug_lvl < 1) {
    return;
  }

  input_layer *input = dynamic_cast<input_layer*>(l);

  if(input->current_root_rank() == 0) {
    if(m->get_comm()->get_rank_in_model() < input->get_data_reader()->get_num_parallel_readers() && !input->is_local_reader_done()) {
      if(m_debug_phase == execution_mode::invalid || m_debug_phase == m->get_execution_mode()) {
        print_fp_start(m, input);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Testing phase debugging
////////////////////////////////////////////////////////////////////////////////
void lbann::lbann_callback_debug_io::on_test_begin(model *m) {
  if(m_debug_phase == execution_mode::invalid || m_debug_phase == execution_mode::testing) {
    print_phase_start(m, execution_mode::testing);
  }
}
