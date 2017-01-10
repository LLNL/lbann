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
// lbann_optimizer .hpp .cpp - Abstract class for neural network optimizers
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN optimizer base class
 * Implements checkpoint and restart
 */
#ifndef LBANN_OPTIMIZER_HPP
#define LBANN_OPTIMIZER_HPP

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/io/lbann_persist.hpp"
#include <vector>
#include <string>

namespace lbann
{
  class Optimizer {
  public:
    Optimizer() {}
    virtual ~Optimizer() {}
    // virtual Optimizer *create_optimizer() {};
    virtual void setup(int input_dims, int num_neurons) {}
    virtual void update_weight_bias_matrix(ElMat &WB_D, ElMat& WB);
    /** Get the optimizer's current learning rate, if any. */
    virtual float get_learning_rate() const { return 0.0f; }
    /** Set the optimizer's learning rate. */
    virtual void set_learning_rate(float _lr) {}
    virtual bool saveToCheckpoint(int fd, const char* filename, uint64_t* bytes) {
      return false;
    }
    virtual bool loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes) {
      return false;
    }
    virtual bool saveToCheckpointShared(persist& p, int Index) {
      return false;
    }
    virtual bool loadFromCheckpointShared(persist& p, int Index) {
      return false;
    }

  public:
  };

  class Optimizer_factory {
  public:
    Optimizer_factory() {}
    virtual ~Optimizer_factory() {}
    virtual Optimizer *create_optimizer(matrix_format format=matrix_format::MC_MR) { return nullptr; };

  };
}

#endif // LBANN_OPTIMIZER_HPP
