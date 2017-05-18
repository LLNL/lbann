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
#include "lbann/callbacks/lbann_callback_dump_minibatch_sample_indices.hpp"

namespace lbann {

void lbann_callback_dump_minibatch_sample_indices::on_forward_prop_end(model* m, Layer* l) {
  const std::string prefix = basename + "model" +
    std::to_string(m->get_comm()->get_model_rank()) +
    "-rank" + std::to_string(m->get_comm()->get_rank_in_model()) +
    "-epoch" + std::to_string(m->get_cur_epoch()) + "-step" +
    std::to_string(m->get_cur_step()) + "-layer";

  if (_layer_type_to_category(l->m_type) != layer_category::io) {
    return;
  }

  El::Matrix<El::Int>& indices = l->get_sample_indices_per_mb();
  El::Write(indices,
            prefix + std::to_string(l->get_index()) +
            "-MB_Sample_Indices",
            El::ASCII);
}

}  // namespace lbann
