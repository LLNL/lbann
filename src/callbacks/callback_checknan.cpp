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
// lbann_callback_checknan .hpp .cpp - Check matrices for invalid numbers
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_checknan.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

void lbann_callback_checknan::on_forward_prop_end(model *m, Layer *l) {
  if (dynamic_cast<target_layer*>(l) != nullptr) {
    return;
  }
  const AbsDistMat& acts = l->get_activations();
  if (!is_good(acts)) {
    lbann_comm *comm = m->get_comm();
    dump_network(m);
    throw lbann_exception(std::string()
      + "checknan: "
      + "[" + std::to_string(comm->get_rank_in_world()) + "]: "
      + "error in layer " + l->get_name() + " "
      + "activations (step=" + std::to_string(m->get_cur_step()) + ")");
  }
}

void lbann_callback_checknan::on_backward_prop_end(model *m, Layer *l) {
  learning *learning_layer = dynamic_cast<learning *>(l);
  if(learning_layer == nullptr) {
    return;
  }
  const AbsDistMat& grad = learning_layer->get_weights_gradient();
  if (!is_good(grad)) {
    lbann_comm *comm = m->get_comm();
    dump_network(m);
    throw lbann_exception(std::string()
      + "checknan: "
      + "[" + std::to_string(comm->get_rank_in_world()) + "]: "
      + "error in layer " + l->get_name() + " "
      + "gradients (step=" + std::to_string(m->get_cur_step()) + ")");
  }
}

void lbann_callback_checknan::on_batch_end(model *m) {
  for (Layer *layer : m->get_layers()) {
    // Skip non-learning layers.
    learning *learning_layer = dynamic_cast<learning *>(layer);
    if(learning_layer == nullptr) {
      continue;
    }
    const AbsDistMat& weights = learning_layer->get_weights();
    if (!is_good(weights)) {
      lbann_comm *comm = m->get_comm();
      dump_network(m);
      throw lbann_exception(std::string()
        + "checknan: "
        + "[" + std::to_string(comm->get_rank_in_world()) + "]: "
        + "error in layer " + layer->get_name() + " "
        + "weights (step=" + std::to_string(m->get_cur_step()) + ")");
    }
  }
}

bool lbann_callback_checknan::is_good(const AbsDistMat& m) {
  const Mat& lm = m.LockedMatrix();
  const Int height = lm.Height();
  const Int width = lm.Width();
  for (Int col = 0; col < width; ++col) {
    for (Int row = 0; row < height; ++row) {
      const DataType val = lm(row, col);
      if (std::isnan(val)) {
        std::cout << "Found NaN at (" << row << ", " << col << ")!" << std::endl;
        return false;
      } else if (std::isinf(val)) {
        std::cout << "Found inf (" << row << ", " << col << ")!" << std::endl;
        return false;
      }
    }
  }
  return true;
}

void lbann_callback_checknan::dump_network(model *m) {
  // Dump only the local matrices because not every rank will necessarily
  // have bad data, and the check is purely local.
  for (Layer *layer : m->get_layers()) {
    const std::string prefix
      = ("model" + std::to_string(m->get_comm()->get_model_rank())
         + "-rank" + std::to_string(m->get_comm()->get_rank_in_model()) +
         + "-epoch" + std::to_string(m->get_cur_epoch())
         + "-step" + std::to_string(m->get_cur_step())
         + "-" + layer->get_name()
         + "-");
    El::Write(layer->get_activations().Matrix(),
              prefix + "Activations",
              El::ASCII);
    learning *learning_layer = dynamic_cast<learning*>(layer);
    if (learning_layer != nullptr) {
      El::Write(learning_layer->get_weights().Matrix(),
                prefix + "WeightsBiases",
                El::ASCII);
      El::Write(learning_layer->get_weights_gradient().Matrix(),
                prefix + "Gradients",
                El::ASCII);
    }
  }
}

}  // namespace lbann
