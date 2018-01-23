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
#include "lbann/layers/io/target/target_layer.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

void lbann_callback_checknan::on_forward_prop_end(model *m, Layer *l) {
  if (dynamic_cast<target_layer*>(l) != nullptr) {
    return;
  }
  const AbsDistMat& acts = l->get_activations();
  if (!is_good(acts)) {
    dump_network(m);
    std::stringstream ss;
    ss << name() << ": "
       << "[" << std::to_string(m->get_comm()->get_rank_in_world()) << "]: "
       << "error in activations of " << l->get_name() << " "
       << "(step=" << std::to_string(m->get_cur_step()) << ")";
    throw lbann_exception(ss.str());
  }
}

void lbann_callback_checknan::on_backward_prop_end(model *m) {
  for (weights *w : m->get_weights()) {
    optimizer *opt = w->get_optimizer();
    if (opt != nullptr && !is_good(opt->get_gradient())) {
      dump_network(m);
      std::stringstream ss;
      ss << name() << ": "
         << "[" << std::to_string(m->get_comm()->get_rank_in_world()) << "]: "
         << "error in weights gradient of " << w->get_name() << " "
         << "(step=" << std::to_string(m->get_cur_step()) << ")";
      throw lbann_exception(ss.str());
    }
  }
}

void lbann_callback_checknan::on_batch_end(model *m) {
  for (weights *w : m->get_weights()) {
    if (!is_good(w->get_values())) {
      dump_network(m);
      std::stringstream ss;
      ss << name() << ": "
         << "[" << std::to_string(m->get_comm()->get_rank_in_world()) << "]: "
         << "error in weights of " << w->get_name() << " "
         << "(step=" << std::to_string(m->get_cur_step()-1) << ")";
      throw lbann_exception(ss.str());
    }
  }
}

bool lbann_callback_checknan::is_good(const AbsDistMat& m) {
  const Mat& lm = m.LockedMatrix();
  const El::Int height = lm.Height();
  const El::Int width = lm.Width();
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
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
    El::Write(layer->get_activations().LockedMatrix(),
              prefix + "Activations",
              El::ASCII);
  }
  for (weights *w : m->get_weights()) {
    const std::string prefix
      = ("model" + std::to_string(m->get_comm()->get_model_rank())
         + "-rank" + std::to_string(m->get_comm()->get_rank_in_model()) +
         + "-epoch" + std::to_string(m->get_cur_epoch())
         + "-step" + std::to_string(m->get_cur_step())
         + "-" + w->get_name()
         + "-");
    El::Write(w->get_values().LockedMatrix(),
              prefix + "Weights",
              El::ASCII);
    optimizer *opt = w->get_optimizer();
    if (opt != nullptr) {
      El::Write(opt->get_gradient().LockedMatrix(),
                prefix + "Gradient",
                El::ASCII);
    }
  }
}

}  // namespace lbann
