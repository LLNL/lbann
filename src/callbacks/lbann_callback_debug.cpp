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

//#include <algorithm>
#include "lbann/callbacks/lbann_callback_debug.hpp"

void lbann::lbann_callback_debug::on_epoch_begin(model* m) {
}

void lbann::lbann_callback_debug::on_epoch_end(model* m) {
}

void lbann::lbann_callback_debug::on_batch_begin(model* m) {
  std::cout << "Phase: " << _to_string(m->get_execution_mode()) << " starting batch" << std::endl;
}

void lbann::lbann_callback_debug::on_batch_end(model* m) {
}

void lbann::lbann_callback_debug::on_forward_prop_begin(model* m, Layer* l) {
  std::cout << "Phase: " << _to_string(m->get_execution_mode()) << " starting forward propagation for layer " << l->Index << std::endl;
}

void lbann::lbann_callback_debug::on_forward_prop_end(model* m, Layer* l) {
  std::cout << "Phase: " << _to_string(m->get_execution_mode()) << "   ending forward propagation for layer " << l->Index << std::endl;
}

void lbann::lbann_callback_debug::on_backward_prop_begin(model* m, Layer* l) {
  std::cout << "Phase: " << _to_string(m->get_execution_mode()) << " starting backward propagation for layer " << l->Index << std::endl;
}

void lbann::lbann_callback_debug::on_backward_prop_end(model* m, Layer* l) {
  std::cout << "Phase: " << _to_string(m->get_execution_mode()) << "   ending backward propagation for layer " << l->Index << std::endl;
}

