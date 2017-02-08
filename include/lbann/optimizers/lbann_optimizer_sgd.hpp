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
/// Inspired by Kera.io implementation
/// Stochastic gradient descent, with support for momentum, decay, and Nesterov momentum.
///  lr: float >= 0. Learning rate.
///  momentum: float >= 0. Parameter updates momentum.
///  decay: float >= 0. Learning rate decay over each update.
///  nesterov: boolean. Whether to apply Nesterov momentum.
//
// lbann_optimizer_sgd .hpp .cpp - Stochastic gradient descent optimizer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_SGD_HPP
#define LBANN_OPTIMIZER_SGD_HPP

#include "lbann/optimizers/lbann_optimizer.hpp"
#include <sys/stat.h>

namespace lbann
{
  template <class _DistMat>
  class SGD : public Optimizer {

  public:
    lbann_comm* comm;
    float lr;
    float momentum;
    float decay;
    bool nesterov;

  private:
    long iterations; // BVE FIXME how do we save / checkpoint this
    _DistMat velocity;
    _DistMat nesterov_ag; // Nesterov Accelerated Gradient term (scratch space only -- do not save / checkpoint)

  public:
    SGD(lbann_comm* comm, float lr, float momentum, float decay, bool nesterov)
      : comm(comm), lr(lr), momentum(momentum),
        decay(decay), nesterov(nesterov),
        velocity(comm->get_model_grid()),
        nesterov_ag(comm->get_model_grid()) {
      set_name("sgd");
      iterations = 0;
    }

    ~SGD() {
      velocity.Empty();
      nesterov_ag.Empty();
    }

    void setup(int input_dim, int num_neurons) {
      if (comm->am_model_master()) {
        printf("Setting up SGD optimizer with velocity size %d x %d - lr =%f momentum=%f decay=%g nesterov=%d\n", num_neurons, input_dim, lr, momentum, decay, nesterov);
      }
      iterations = 0;
      Zeros(velocity, num_neurons, input_dim);
      Zeros(nesterov_ag, num_neurons, input_dim);
    }

    void update_weight_bias_matrix(ElMat& WB_D, ElMat& WB) {

      // KERAS: lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
      lr = lr * (1.0 / (1.0 + decay * iterations));
      iterations++;
  
      // velocity
      // KERAS: v = self.momentum * m - lr * g  # velocity
      Scale(momentum, velocity);
      Scale(lr, WB_D); // step
      Axpy(-1, WB_D.Matrix(), velocity.Matrix()); // Update // TODO

      if(nesterov) {
        //KERAS: new_p = p + self.momentum * v - lr * g
        Copy(velocity, nesterov_ag);
        Scale(momentum, nesterov_ag);
        Axpy(-1, WB_D.Matrix(), nesterov_ag.Matrix()); // TODO
        Axpy(1., nesterov_ag.Matrix(), WB.Matrix()); // TODO
      }else {
        //KERAS: new_p = p + v
        Axpy(1., velocity.Matrix(), WB.Matrix()); // TODO
      }

    }

    float get_learning_rate() const { return lr; }
    void set_learning_rate(float _lr) { lr = _lr; }

    bool saveToCheckpoint(int fd, const char* filename, uint64_t* bytes) {
      //    writeDist(fd, filename, velocity, bytes);
      return true;
    }

    bool loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes) {
      //    readDist(fd, filename, velocity, bytes);
      return true;
    }

    bool saveToCheckpointShared(persist& p, int Index) {
      char name[512];

      // current learning rate value
      if (p.m_rank == 0) {
        sprintf(name, "L%d_learning_rate", Index);
        p.write_float(persist_type::train, name, lr);
      }

      // build name of the checkpoint file
      sprintf(name, "L%d_sgd_%lldx%lld", Index, velocity.Height(), velocity.Width());
      p.write_distmat(persist_type::train, name, (DistMat*)&velocity);

      return true;
    }

    bool loadFromCheckpointShared(persist& p, int Index) {
      char name[512];

      // current learning rate value
      if (p.m_rank == 0) {
        sprintf(name, "L%d_learning_rate", Index);
        p.read_float(persist_type::train, name, &lr);
      }
      MPI_Bcast(&lr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

      // build name of the checkpoint file
      sprintf(name, "L%d_sgd_%lldx%lld.bin", Index, velocity.Height(), velocity.Width());
      p.read_distmat(persist_type::train, name, (DistMat*)&velocity);

      return true;
    }
    
  };

  class SGD_factory : public Optimizer_factory {
  public:
    SGD_factory(lbann_comm* comm, float lr=0.01, float momentum=0.3, float decay=0.0, bool nesterov=false);
    ~SGD_factory();
    Optimizer *create_optimizer(matrix_format format=matrix_format::MC_MR);
    const string name() { return "sgd"; }
  public:
    lbann_comm* comm;
    float lr;
    float momentum;
    float decay;
    bool nesterov;
  };
}

#endif // LBANN_OPTIMIZER_SGD_HPP
