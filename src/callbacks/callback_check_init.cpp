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
// lbann_callback_check_init .hpp .cpp - Check multi-model init
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_check_init.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

void lbann_callback_check_init::on_train_begin(model *m) {
  // Skip after the first epoch.
  if (m->get_cur_epoch() != 0) {
    return;
  }
  lbann_comm *comm = m->get_comm();
  if (comm->am_world_master()) {
    std::cout << "Checking all model initial weights match..." << std::endl;
  }
  if (comm->get_num_models() == 1) {
    return;
  }

  std::vector<Layer *>& layers = m->get_layers();
  // Skip the input/output layers.
  for (size_t l = 1; l < layers.size() - 1; ++l) {
    if (comm->am_world_master()) {
      std::cout << "Checking layer " << l << std::endl;
    }
    // Skip non-learning layers.
    learning *learning_layer = (learning *) dynamic_cast<learning *> (layers[l]);
    if(learning_layer == NULL) {
      continue;
    }
    // Model 0 holds the master copy, it gathers the values from other models
    // and compares them.
    const AbsDistMat& weights = learning_layer->get_weights();
    const Mat& local_weights = weights.LockedMatrix();
    for (int model = 1; model < comm->get_num_models(); ++model) {
      comm->global_barrier();
      if (comm->get_model_rank() == 0) {
        Mat remote_weights(local_weights.Height(), local_weights.Width());
        comm->recv(remote_weights, model);
        if (!check_equal(local_weights, remote_weights)) {
          throw lbann_exception(
            "check_init: model " + std::to_string(model) + " rank in model " +
            std::to_string(comm->get_rank_in_model()) +
            " does not match model 0");
        }
      } else if (comm->get_model_rank() == model) {
        comm->send(local_weights, 0);
      }
    }
  }
}

bool lbann_callback_check_init::check_equal(const Mat& x, const Mat& y) const {
  const Int height = x.Height();
  const Int width = x.Width();
  if (height != y.Height() || width != y.Width() || x.LDim() != y.LDim()) {
    return false;
  }
  const DataType *x_buf = x.LockedBuffer();
  const DataType *y_buf = y.LockedBuffer();
  for (Int i = 0; i < height * width; ++i) {
    if (x_buf[i] != y_buf[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace lbann
