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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_checksmall.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

void lbann_callback_checksmall::on_forward_prop_end(model *m, Layer *l) {
  const AbsDistMat& acts = l->get_activations();
  if (!is_good(acts)) {
    std::stringstream ss;
    ss << name() << ": "
       << "[" << std::to_string(m->get_comm()->get_rank_in_world()) << "]: "
       << "error in activations of " << l->get_name() << " "
       << "(step=" << std::to_string(m->get_step(execution_mode::training)) << ")";
    throw lbann_exception(ss.str());
  }
}

void lbann_callback_checksmall::on_backward_prop_end(model *m) {
  for (weights *w : m->get_weights()) {
    optimizer *opt = w->get_optimizer();
    if (opt != nullptr && !is_good(opt->get_gradient())) {
      std::stringstream ss;
      ss << name() << ": "
         << "[" << std::to_string(m->get_comm()->get_rank_in_world()) << "]: "
         << "error in weights gradient of " << w->get_name() << " "
         << "(step=" << std::to_string(m->get_step(execution_mode::training)) << ")";
      throw lbann_exception(ss.str());
    }
  }
}

void lbann_callback_checksmall::on_batch_end(model *m) {
  for (weights *w : m->get_weights()) {
    if (!is_good(w->get_values())) {
      std::stringstream ss;
      ss << name() << ": "
         << "[" << std::to_string(m->get_comm()->get_rank_in_world()) << "]: "
         << "error in weights of " << w->get_name() << " "
         << "(step=" << std::to_string(m->get_step(execution_mode::training)-1) << ")";
      throw lbann_exception(ss.str());
    }
  }
}

bool lbann_callback_checksmall::is_good(const AbsDistMat& m) {
  const AbsMat& local_mat = m.LockedMatrix();
  const El::Int height = local_mat.Height();
  const El::Int width = local_mat.Width();
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const DataType val = std::abs(local_mat(row, col));
      if (val > 0 && val <= m_threshold) {
        std::cout << "Found small value " << val << " "
                  << "at (" << row << "," << col << ")!" << std::endl;
        return false;
      }
    }
  }
  return true;
}

}  // namespace lbann
