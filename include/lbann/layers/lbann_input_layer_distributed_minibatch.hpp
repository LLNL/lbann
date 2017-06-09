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

#ifndef LBANN_LAYERS_INPUT_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED

#include "lbann/layers/lbann_input_layer.hpp"

namespace lbann {
class input_layer_distributed_minibatch : public input_layer {
 public:
  input_layer_distributed_minibatch(data_layout data_dist, lbann_comm *comm, uint mini_batch_size, std::map<execution_mode, DataReader *> data_readers, std::vector<regularizer *> regs= {});

  void setup(int num_prev_neurons);
  bool update();
  Mat *get_local_mat() {
    return &X_local;
  };
  CircMat *get_dist_mat() {
    return &Xs;
  };

 public:
  int m_root; /* Which rank is the root of the CircMat */
  Mat X_local;
  CircMat Xs;
  long m_num_data_per_epoch;
 protected:
  /** Handle forward propagation (arguments are unused.) */
  void fp_linearity();
};
}

#endif  // LBANN_LAYERS_INPUT_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
