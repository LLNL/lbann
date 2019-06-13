////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/callbacks/callback_checknan.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

namespace {

/** Check whether a matrix contains a NaN.
 *  If a NaN entry is detected, return true and output the local entry
 *  position in row and col. mat is assumed to be a CPU matrix.
 */
bool has_nan(const AbsDistMat& mat, El::Int& row, El::Int& col) {
  row = -1;
  col = -1;
  const auto& local_mat = mat.LockedMatrix();
  for (El::Int j = 0; j < local_mat.Width(); ++j) {
    for (El::Int i = 0; i < local_mat.Height(); ++i) {
      if (std::isnan(local_mat(i,j))) {
        row = i;
        col = j;
        return true;
      }
    }
  }
  return false;
}

/** Check whether a matrix contains an inf.
 *  If an inf entry is detected, return true and output the entry
 *  position in row and col. mat is assumed to be a CPU matrix.
 */
bool has_inf(const AbsDistMat& mat, El::Int& row, El::Int& col) {
  row = -1;
  col = -1;
  const auto& local_mat = mat.LockedMatrix();
  for (El::Int j = 0; j < local_mat.Width(); ++j) {
    for (El::Int i = 0; i < local_mat.Height(); ++i) {
      if (std::isinf(local_mat(i,j))) {
        row = i;
        col = j;
        return true;
      }
    }
  }
  return false;
}

/** Dump the local network matrices for debugging.
 *  Dump only the local matrices because not every rank will
 *  necessarily have bad data, and the check is purely local.
 */
void dump_network(model *m) {
  for (const auto* l : m->get_layers()) {
    std::stringstream ss;
    ss << "model" << m->get_comm()->get_trainer_rank()
       << "-rank" << m->get_comm()->get_rank_in_trainer()
       << "-epoch" << m->get_epoch()
       << "-step" << m->get_step(execution_mode::training)
       << "-" << l->get_name() << "-";
    const std::string prefix = ss.str();
    for (int i = 0; i < l->get_num_children(); ++i) {
      El::Write(l->get_local_activations(i),
                prefix + "Activations" + std::to_string(i),
                El::ASCII);
    }
    for (int i = 0; i < l->get_num_parents(); ++i) {
      El::Write(l->get_local_error_signals(i),
                prefix + "ErrorSignal" + std::to_string(i),
                El::ASCII);
    }
  }
  for (auto* w : m->get_weights()) {
    std::stringstream ss;
    ss << "model" << m->get_comm()->get_trainer_rank()
       << "-rank" << m->get_comm()->get_rank_in_trainer()
       << "-epoch" << m->get_epoch()
       << "-step" << m->get_step(execution_mode::training)
       << "-" << w->get_name() << "-";
    const std::string prefix = ss.str();
    El::Write(w->get_values().LockedMatrix(),
              prefix + "Weights",
              El::ASCII);
    auto* opt = w->get_optimizer();
    if (opt != nullptr) {
      El::Write(opt->get_gradient().LockedMatrix(),
                prefix + "Gradient",
                El::ASCII);
    }
  }
}

} // namespace

void lbann_callback_checknan::on_forward_prop_end(model *m, Layer *l) {
  std::stringstream err;
  const auto& num_outputs = l->get_num_children();
  for (int i = 0; i < num_outputs; ++i) {
    El::Int row, col;
    AbsDistMatReadProxy<El::Device::CPU> mat_proxy(l->get_activations(i));
    if (has_nan(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      err << "rank " << m->get_comm()->get_rank_in_world() << ": "
          << "local entry (" << row << "," << col << ") is NaN "
          << "in activations ";
      if (num_outputs > 1) { err << i << " "; }
      err << "of layer \"" << l->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
    if (has_inf(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      err << "rank " << m->get_comm()->get_rank_in_world() << ": "
          << "local entry (" << row << "," << col << ") is inf "
          << "in activations ";
      if (num_outputs > 1) { err << i << " "; }
      err << "of layer \"" << l->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
  }
}

void lbann_callback_checknan::on_backward_prop_end(model *m, Layer *l) {
  std::stringstream err;
  const auto& num_inputs = l->get_num_parents();
  for (int i = 0; i < num_inputs; ++i) {
    El::Int row, col;
    AbsDistMatReadProxy<El::Device::CPU> mat_proxy(l->get_error_signals(i));
    if (has_nan(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      err << "rank " << m->get_comm()->get_rank_in_world() << ": "
          << "local entry (" << row << "," << col << ") is NaN "
          << "in error signals ";
      if (num_inputs > 1) { err << i << " "; }
      err << "of layer \"" << l->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
    if (has_inf(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      err << "rank " << m->get_comm()->get_rank_in_world() << ": "
          << "local entry (" << row << "," << col << ") is inf "
          << "in error signals ";
      if (num_inputs > 1) { err << i << " "; }
      err << "of layer \"" << l->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
  }
}

void lbann_callback_checknan::on_backward_prop_end(model *m) {
  std::stringstream err;
  for (weights *w : m->get_weights()) {
    auto* opt = w->get_optimizer();
    if (opt != nullptr) {
      El::Int row, col;
      AbsDistMatReadProxy<El::Device::CPU> mat_proxy(opt->get_gradient());
      if (has_nan(mat_proxy.GetLocked(), row, col)) {
        dump_network(m);
        err << "rank " << m->get_comm()->get_rank_in_world() << ": "
            << "local entry (" << row << "," << col << ") is NaN "
            << "in gradient w.r.t. weights \"" << w->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
      if (has_inf(mat_proxy.GetLocked(), row, col)) {
        dump_network(m);
        err << "rank " << m->get_comm()->get_rank_in_world() << ": "
            << "local entry (" << row << "," << col << ") is inf "
            << "in gradient w.r.t. weights \"" << w->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
    }
  }
}

void lbann_callback_checknan::on_batch_end(model *m) {
  std::stringstream err;
  for (weights *w : m->get_weights()) {
    El::Int row, col;
    AbsDistMatReadProxy<El::Device::CPU> mat_proxy(w->get_values());
    if (has_nan(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      err << "rank " << m->get_comm()->get_rank_in_world() << ": "
          << "local entry (" << row << "," << col << ") is NaN "
          << "in weights \"" << w->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
    if (has_inf(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      err << "rank " << m->get_comm()->get_rank_in_world() << ": "
          << "local entry (" << row << "," << col << ") is inf "
          << "in weights \"" << w->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
  }
}

}  // namespace lbann
