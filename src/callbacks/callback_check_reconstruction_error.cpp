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
// lbann_check_reconstruction_error .hpp .cpp - Callback hooks for termination 
// after reconstruction error has reached a given value
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_check_reconstruction_error.hpp"

namespace lbann {

lbann_callback_check_reconstruction_error::lbann_callback_check_reconstruction_error(double max_error) :
  lbann_callback(), m_max_error(max_error) {}


void lbann_callback_check_reconstruction_error::on_epoch_end(model *m) {
  double reconstr_error  = m->m_obj_fn->get_mean_value();
  if( reconstr_error < m_max_error) {
    if (m->get_comm()->am_model_master()) {
      std::cout << "Reconstruction error " << reconstr_error << "is less than " <<  m_max_error << 
        " TEST PASSED" << std::endl;
    }
     exit(0);
  }
}

}  // namespace lbann
