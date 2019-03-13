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
// lbann_callback_dump_minibatch_sample_indices .hpp .cpp - Callbacks
// to dump the list of indices per minibatch
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include "lbann/callbacks/callback_dump_minibatch_sample_indices.hpp"
#include "lbann/layers/io/input/input_layer.hpp"
#include <iomanip>
#include <cstdlib>

namespace lbann {

void lbann_callback_dump_minibatch_sample_indices::dump_to_file(model *m, Layer *l, int64_t step) {
  // Print minibatch sample indices of input layers
  auto *input = dynamic_cast<generic_input_layer*>(l);
  if (input != nullptr) {
    El::Matrix<El::Int>* indices = l->get_sample_indices_per_mb();
    if (indices == nullptr
        || indices->Height() == 0
        || indices->Width() == 0) {
      return;
    }

    std::ostringstream s;
    s << "mkdir -p " << m_basename;
    const int dir= system(s.str().c_str());
    if (dir< 0) {
      LBANN_ERROR("callback_dump_minibatch_sample_indices is unable to create the target director");
    }

    const std::string file
      = (m_basename
         + _to_string(m->get_execution_mode())
         + "-model" + std::to_string(m->get_comm()->get_trainer_rank())
         + "-rank" + std::to_string(m->get_comm()->get_rank_in_trainer())
         + "-epoch" + std::to_string(m->get_epoch())
         + "-step" + std::to_string(m->get_step(execution_mode::training))
         + "-" + l->get_name()
         + "-MB_Sample_Indices");
    El::Write(*indices, file, El::ASCII);
  }
}

void lbann_callback_dump_minibatch_sample_indices::on_forward_prop_end(model *m, Layer *l) {
  dump_to_file(m, l, m->get_step());
}

void lbann_callback_dump_minibatch_sample_indices::on_evaluate_forward_prop_end(model *m, Layer *l) {
  dump_to_file(m, l, m->get_step());
}

}  // namespace lbann
