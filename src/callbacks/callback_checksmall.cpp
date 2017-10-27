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
// lbann_callback_checksmall .hpp .cpp - Check matrices for small values
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_checksmall.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

void lbann_callback_checksmall::on_forward_prop_end(model *m, Layer *l) {
  if (dynamic_cast<target_layer*>(l) != nullptr) {
    return;
  }
  const AbsDistMat& acts = l->get_activations();
  if (!is_good(acts)) {
    lbann_comm *comm = m->get_comm();
    throw lbann_exception(std::string()
      + "checksmall: "
      + "[" + std::to_string(comm->get_rank_in_world()) + "]: "
      + "error in layer " + l->get_name() + " "
      + "activations (step=" + std::to_string(m->get_cur_step()) + ")");
  }
}

void lbann_callback_checksmall::on_backward_prop_end(model *m, Layer *l) {
  learning *learning_layer = dynamic_cast<learning *> (l);
  if(learning_layer == nullptr) {
    return;
  }
  const AbsDistMat& grad = learning_layer->get_weights_gradient();
  if (!is_good(grad)) {
    lbann_comm *comm = m->get_comm();
    throw lbann_exception(std::string()
      + "checksmall: "
      + "[" + std::to_string(comm->get_rank_in_world()) + "]: "
      + "error in layer " + l->get_name() + " "
      + "gradients ("
      + "shape=" + std::to_string(grad.Height()) + "x" + std::to_string(grad.Width()) + " "
      + "step=" + std::to_string(m->get_cur_step()) + ")");
  }
}

void lbann_callback_checksmall::on_batch_end(model *m) {
  for (Layer *layer : m->get_layers()) {
    // Skip non-learning layers.
    learning *learning_layer = dynamic_cast<learning *>(layer);
    if(learning_layer == nullptr) {
      continue;
    }
    const AbsDistMat& weights = learning_layer->get_weights();
    if (!is_good(weights)) {
      lbann_comm *comm = m->get_comm();
      throw lbann_exception(std::string()
        + "checksmall: "
        + "[" + std::to_string(comm->get_rank_in_world()) + "]: "
        + "error in layer " + layer->get_name() + " "
        + "weights (step=" + std::to_string(m->get_cur_step()) + ")");
    }
  }
}

bool lbann_callback_checksmall::is_good(const AbsDistMat& m) {
  const Mat& local_mat = m.LockedMatrix();
  const Int height = local_mat.Height();
  const Int width = local_mat.Width();
  for (Int col = 0; col < width; ++col) {
    for (Int row = 0; row < height; ++row) {
      const DataType val = Abs(local_mat(row, col));
      if (val > 0 && val <= m_threshold) {
        std::cout << "Found small value " << val << " at (" << row << "," <<
                  col << ")!" << std::endl;
        return false;
      }
    }
  }
  return true;
}

}  // namespace lbann
