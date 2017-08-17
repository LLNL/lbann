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
  // Skip output layer.
  if (l->get_index() == (int) m->get_layers().size() - 1) {
    return;
  }
  DistMat& acts = (DistMat&) l->get_activations();
  if (!is_good(acts)) {
    lbann_comm *comm = m->get_comm();
    dump_network(m);
    throw lbann_exception(
      "checknan: [" + std::to_string(comm->get_rank_in_world()) +
      "]: error in layer " + std::to_string(l->get_index()) +
      " gradients (step=" + std::to_string(m->get_cur_step()) + ")");
  }
}

void lbann_callback_checknan::on_backward_prop_end(model *m, Layer *l) {
  // Skip non-learning layers.
  learning *learning_layer = (learning *) dynamic_cast<learning *> (l);
  if(learning_layer == NULL) {
    return;
  }
  // Skip input/output layers.
  if (l->get_index() == 0 || l->get_index() == (int) m->get_layers().size() - 1) {
    return;
  }
  DistMat& grad = (DistMat&) learning_layer->get_weights_biases_gradient();
  if (!is_good(grad)) {
    lbann_comm *comm = m->get_comm();
    dump_network(m);
    throw lbann_exception(
      "checknan: [" + std::to_string(comm->get_rank_in_world()) +
      "]: error in layer " + std::to_string(l->get_index()) +
      " gradients (step=" + std::to_string(m->get_cur_step()) + ")");
  }
}

void lbann_callback_checknan::on_batch_end(model *m) {
  std::vector<Layer *>& layers = m->get_layers();
  // Skip input/output layers-- they don't have weights.
  for (size_t i = 1; i < layers.size() - 1; ++i) {
    Layer *l = layers[i];
    // Skip non-learning layers.
    learning *learning_layer = (learning *) dynamic_cast<learning *> (l);
    if(learning_layer == NULL) {
      continue;
    }
    DistMat& weights = (DistMat&) learning_layer->get_weights_biases();
    if (!is_good(weights)) {
      lbann_comm *comm = m->get_comm();
      dump_network(m);
      throw lbann_exception(
      "checknan: [" + std::to_string(comm->get_rank_in_world()) +
      "]: error in layer " + std::to_string(l->get_index()) +
      " weights (step=" + std::to_string(m->get_cur_step()) + ")");
    }
  }
}

bool lbann_callback_checknan::is_good(const DistMat& m) {
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
  std::vector<Layer*>& layers = m->get_layers();
  // Dump only the local matrices because not every rank will necessarily
  // have bad data, and the check is purely local.
  const std::string prefix =
    "model" + std::to_string(m->get_comm()->get_model_rank()) +
    "-rank" + std::to_string(m->get_comm()->get_rank_in_model()) +
    "-epoch" + std::to_string(m->get_cur_epoch()) + "-step" +
    std::to_string(m->get_cur_step()) + "-layer";
  for (size_t idx = 0; idx < layers.size(); ++idx) {
    Layer *l = layers[idx];
    // Dump activations.
    El::Write(l->get_activations().Matrix(),
              prefix + std::to_string(l->get_index()) + "-Activations",
              El::ASCII);
    learning *learning_layer = dynamic_cast<learning*>(l);
    if (learning_layer != NULL) {
      El::Write(learning_layer->get_weights_biases().Matrix(),
                prefix + std::to_string(l->get_index()) + "-WeightsBiases",
                El::ASCII);
      El::Write(learning_layer->get_weights_biases_gradient().Matrix(),
                prefix + std::to_string(l->get_index()) + "-Gradients",
                El::ASCII);
    }
  }
}

}  // namespace lbann
