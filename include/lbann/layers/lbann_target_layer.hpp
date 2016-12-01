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

#ifndef LBANN_LAYERS_TARGET_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_TARGET_LAYER_HPP_INCLUDED

#include "lbann/layers/lbann_io_layer.hpp"

namespace lbann
{
  class target_layer : public io_layer {
  public:
    target_layer(lbann_comm* comm, uint mini_batch_size, std::map<execution_mode, DataReader*> data_readers, bool shared_data_reader, bool for_regression=false);
    DistMat *fp_output();
    DataReader *set_training_data_reader(DataReader *data_reader, bool shared_data_reader);
    DataReader *set_testing_data_reader(DataReader *data_reader, bool shared_data_reader);

    void setup(int num_prev_neurons);
    void fp_set_std_matrix_view();
    /** No non-linearity */
    void fp_nonlinearity() {}
    /** No non-linearity */
    void bp_nonlinearity() {}

    DataType compute_cost_cross_entropy();

    void summarize(lbann_summary& summarizer, int64_t step);
    void epoch_print() const;
    void epoch_reset();
    void resetCost();
    DataType avgCost() const;

    bool saveToCheckpoint(int fd, const char* filename, uint64_t* bytes);
    bool loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes);

    bool saveToCheckpointShared(const char* dir, uint64_t* bytes);
    bool loadFromCheckpointShared(const char* dir, uint64_t* bytes);

  public:
    bool m_shared_data_reader;

  protected:
    DataType aggregate_cost;   // if this type is changed, update checkpoint code
    long num_backprop_steps; // if this type is changed, update checkpoint code
    DistMat m_activations_cost;
    DistMat m_activations_cost_v;
    /** Colume-wise sum of the costs of a minibatch. */
    ColSumMat m_minibatch_cost;
  };
}

#endif  // LBANN_LAYERS_TARGET_LAYER_HPP_INCLUDED
