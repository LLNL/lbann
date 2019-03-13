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
// lbann_callback_dump_gradients .hpp .cpp - Callbacks to dump gradients
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include "lbann/callbacks/callback_dump_gradients.hpp"

namespace lbann {

void lbann_callback_dump_gradients::on_backward_prop_end(model *m) {
  for (weights *w : m->get_weights()) {
    optimizer *opt = w->get_optimizer();
    if (opt != nullptr) {
      const std::string file
        = (m_basename
           + "model" + std::to_string(m->get_comm()->get_trainer_rank())
           + "-epoch" + std::to_string(m->get_epoch())
           + "-step" + std::to_string(m->get_step())
           + "-" + w->get_name()
           + "-Gradient");
      El::Write(opt->get_gradient(), file, El::ASCII);
    }
  }
}

}  // namespace lbann
