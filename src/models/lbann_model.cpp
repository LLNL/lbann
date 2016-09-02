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
// lbann_model .hpp .cpp - Abstract class for neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/lbann_model.hpp"
#include "lbann/callbacks/lbann_callback.hpp"
#include <string>

using namespace std;
using namespace El;

lbann::model::model(lbann_comm* comm) :
  m_execution_mode(execution_mode::invalid),
  m_terminate_training(false),
  m_current_epoch(0), m_current_step(0),
  m_current_mini_batch_size(0),
  comm(comm) {}

void lbann::model::add_callback(lbann::lbann_callback* cb) {
  callbacks.push_back(cb);
}

void lbann::model::setup_callbacks() {
  for (auto&& cb : callbacks) {
    cb->setup(this);
  }
}

void lbann::model::do_train_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_train_begin(this);
  }
}

void lbann::model::do_train_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_train_end(this);
  }
}

void lbann::model::do_epoch_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_epoch_begin(this);
  }
}

void lbann::model::do_epoch_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_epoch_end(this);
  }
}

void lbann::model::do_batch_begin_cbs() {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_batch_begin(this);
    }
  }
}

void lbann::model::do_batch_end_cbs() {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_batch_end(this);
    }
  }
}

void lbann::model::do_test_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_test_begin(this);
  }
}

void lbann::model::do_test_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_test_end(this);
  }
}

void lbann::model::do_validation_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_validation_begin(this);
  }
}

void lbann::model::do_validation_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_validation_end(this);
  }
}

void lbann::model::do_model_forward_prop_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_forward_prop_begin(this);
  }
}

void lbann::model::do_layer_forward_prop_begin_cbs(Layer* l) {
  for (auto&& cb : callbacks) {
    cb->on_forward_prop_begin(this, l);
  }
}

void lbann::model::do_model_forward_prop_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_forward_prop_end(this);
  }
}

void lbann::model::do_layer_forward_prop_end_cbs(Layer* l) {
  for (auto&& cb : callbacks) {
    cb->on_forward_prop_end(this, l);
  }
}

void lbann::model::do_model_backward_prop_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_backward_prop_begin(this);
  }
}

void lbann::model::do_layer_backward_prop_begin_cbs(Layer* l) {
  for (auto&& cb : callbacks) {
    cb->on_backward_prop_begin(this, l);
  }
}

void lbann::model::do_model_backward_prop_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_backward_prop_end(this);
  }
}

void lbann::model::do_layer_backward_prop_end_cbs(Layer* l) {
  for (auto&& cb : callbacks) {
    cb->on_backward_prop_end(this, l);
  }
}
