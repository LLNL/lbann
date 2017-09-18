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
  // Skip output layer.
  if (l->get_index() == (int) m->get_layers().size() - 1) {
    return;
  }
  AbsDistMat& acts = l->get_activations();
  if (!is_good(acts)) {
    lbann_comm *comm = m->get_comm();
    std::cout << "[" << comm->get_rank_in_world() << "]: error in layer " <<
              l->get_index() << " activations (step=" << m->get_cur_step() << ")" <<
              std::endl;
    throw lbann_exception("checksmall: error in activations");
  }
}

void lbann_callback_checksmall::on_backward_prop_end(model *m, Layer *l) {
  // Skip input/output layers.
  if (l->get_index() == 0 || l->get_index() == (int) m->get_layers().size() - 1) {
    return;
  }
  // Skip non-learning layers.
  learning *learning_layer = (learning *) dynamic_cast<learning *> (l);
  if(learning_layer == NULL) {
    return;
  }

  AbsDistMat& grad = learning_layer->get_weights_gradient();
  if (!is_good(grad)) {
    lbann_comm *comm = m->get_comm();
    std::cout << "[" << comm->get_rank_in_world() << "]: error in layer " <<
              l->get_index() << " gradients (shape=" << grad.Height() << "x" <<
              grad.Width() << " step=" << m->get_cur_step() << ")" <<
              std::endl;
    throw lbann_exception("checksmall: error in gradients");
  }
}

void lbann_callback_checksmall::on_batch_end(model *m) {
  std::vector<Layer *>& layers = m->get_layers();
  // Skip input/output layers-- they don't have weights.
  for (size_t i = 1; i < layers.size() - 1; ++i) {
    Layer *l = layers[i];
    // Skip non-learning layers.
    learning *learning_layer = (learning *) dynamic_cast<learning *> (l);
    if(learning_layer == NULL) {
      continue;
    }
    AbsDistMat& weights = learning_layer->get_weights();
    if (!is_good(weights)) {
      lbann_comm *comm = m->get_comm();
      std::cout << "[" << comm->get_rank_in_world() << "]: error in layer " <<
                l->get_index() << " weights (step=" << m->get_cur_step() << ")" <<
                std::endl;
      throw lbann_exception("checksmall: error in weights");
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
