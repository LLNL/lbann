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
#include <iomanip>

namespace lbann {

void lbann_callback_dump_minibatch_sample_indices::dump_to_file(model *m, Layer *l, int64_t step) {
  const std::string prefix = m_basename + _to_string(m->get_execution_mode()) + "-model" +
                             std::to_string(m->get_comm()->get_model_rank()) +
                             "-rank" + std::to_string(m->get_comm()->get_rank_in_model()) +
                             "-epoch" + std::to_string(m->get_cur_epoch()) + "-step" +
                             std::to_string(step) + "-layer";
  if (!dynamic_cast<io_layer*>(l) || l->get_index() != 0) {
    return;
  }

  El::Matrix<El::Int>* indices = l->get_sample_indices_per_mb();

  if(indices->Height() != 0 && indices->Width() != 0) {
    El::Write(*indices,
              prefix + std::to_string(l->get_index()) +
              "-MB_Sample_Indices",
              El::ASCII);
  }
}

void lbann_callback_dump_minibatch_sample_indices::on_forward_prop_end(model *m, Layer *l) {
  dump_to_file(m, l, m->get_cur_step());
}

void lbann_callback_dump_minibatch_sample_indices::on_evaluate_forward_prop_end(model *m, Layer *l) {
  switch(m->get_execution_mode()) {
  case execution_mode::validation:
    dump_to_file(m, l, m->get_cur_validation_step());
    break;
  case execution_mode::testing:
    dump_to_file(m, l, m->get_cur_testing_step());
    break;
  default:
    throw lbann_exception("lbann_callback_dump_minibatch_sample_indices: invalid execution phase");
  }
}

}  // namespace lbann
