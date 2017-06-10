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

#ifndef LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED
#define LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/layers/lbann_target_layer.hpp"

namespace lbann {
class reconstruction_layer : public target_layer {
 public:
  reconstruction_layer(data_layout data_dist, size_t index,lbann_comm *comm,
                       optimizer *optimizer,
                       const uint minim_batch_size,
                       Layer *original_layer,
                       activation_type activation=activation_type::RELU,
                       weight_initialization init=weight_initialization::glorot_uniform);

  void setup(int num_prev_neurons);
  bool update(void);
  void summarize(lbann_summary& summarizer, int64_t step);
  void epoch_print(void) const;
  void epoch_reset(void);
  execution_mode get_execution_mode(void);
  DataType reconstruction_cost(const DistMat& Y);
  void reset_cost(void);
  DataType average_cost(void) const;


 protected:
  void fp_linearity(void);
  void bp_linearity(void);
  void fp_nonlinearity(void);
  void bp_nonlinearity(void);

 private:
  Layer *m_original_layer;
  DataType aggregate_cost;
  long num_forwardprop_steps;
  weight_initialization m_weight_initialization;
};
}

#endif  // LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED
