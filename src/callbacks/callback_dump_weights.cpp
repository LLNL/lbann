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
// lbann_callback_dump_weights .hpp .cpp - Callbacks to dump weight matrices
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include "lbann/callbacks/callback_dump_weights.hpp"

namespace lbann {

void lbann_callback_dump_weights::on_train_begin(model *m) {
  dump_weights(m, "initial");
}

void lbann_callback_dump_weights::on_epoch_end(model *m) {
  dump_weights(m);
}

void lbann_callback_dump_weights::dump_weights(model *m, std::string s) {
  for (weights *w : m->get_weights()) {
    std::string epoch = "-epoch" + std::to_string(m->get_epoch()-1);
    if(s != "") {
      epoch = "-" + s;
    }
    const std::string file
      = (m_basename
         + "model" + std::to_string(m->get_comm()->get_trainer_rank())
         + epoch
         + "-" + w->get_name()
         + "-Weights");
    El::Write(w->get_values(), file, El::ASCII);
  }
}

}  // namespace lbann
