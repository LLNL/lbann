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
// lbann_optimizer_adagrad .hpp .cpp - Stochastic gradient descent optimizer with Adagrad
//
// Inspired by Kera.io implementation
//
/// Stochastic gradient descent with Adagrad.
///  lr: float >= 0. Learning rate.
///  epsilon: float >= 0.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_ADAGRAD_HPP
#define LBANN_OPTIMIZER_ADAGRAD_HPP

#include "lbann/optimizers/lbann_optimizer.hpp"
#include <sys/stat.h>

namespace lbann
{
  template<class _DistMat>
  class Adagrad : public Optimizer {

  public:
    float lr;

    float epsilon;

    lbann_comm* comm;
    _DistMat     WB_D_Cache;     // Cache of Weights and Bias Gradient (current time t - 1)
    _DistMat     WB_D_Temp;      // Temporary for Weights and Bias Gradient computation
    _DistMat     WB_D_Temp2;     // Temporary for Weights and Bias Gradient computation

  private:
    static inline DataType _sq(const DataType& x) { return (x * x); }
    static inline DataType _sqrt(const DataType& x) { return (1 / sqrt(x + 1e-8)); }

  public:

    /// Constructor
    Adagrad(lbann_comm* comm, float lr, float epsilon)
      : lr(lr), epsilon(epsilon), comm(comm),
        WB_D_Cache(comm->get_model_grid()),
        WB_D_Temp(comm->get_model_grid()),
        WB_D_Temp2(comm->get_model_grid()) {
      set_name("adagrad");
      if (comm->am_model_master()) {
        printf("Initializing Adagrad optimizer with lr=%f and epsilon=%f\n", lr, epsilon);
      }
    }

    /// Destructor
    ~Adagrad() {
      WB_D_Cache.Empty();
      WB_D_Temp.Empty();
      WB_D_Temp2.Empty();
    }

    /// Setup optimizer
    void setup(int input_dim, int num_neurons) {
      if (comm->am_model_master()) {
        printf("Setting up Adagrad optimizer with cache size %d x %d\n", num_neurons, input_dim);
      }
      Zeros(WB_D_Cache, num_neurons, input_dim);
      Zeros(WB_D_Temp, num_neurons, input_dim);
      Zeros(WB_D_Temp2, num_neurons, input_dim);  
      if (comm->am_model_master()) {
        printf("Setting up Adagrad optimizer with WB_D_Cache size %d x %d\n", WB_D_Cache.Height(), WB_D_Cache.Width());  
      }
    }
    
    void update_weight_bias_matrix(ElMat& WB_D, ElMat& WB) {
      Copy(WB_D, WB_D_Temp);
      // Square each entry of the WB_D matrix
      EntrywiseMap(WB_D_Temp, std::function<DataType(const DataType&)>(_sq));
      // Add squared value to WB_D_Cache
      Axpy(1., WB_D_Temp, WB_D_Cache);

      Copy(WB_D_Cache, WB_D_Temp);
      // Compute the inverse of the square root of the historical gradient (with a small perturbation)
      EntrywiseMap(WB_D_Temp, std::function<DataType(const DataType&)>(_sqrt));
      Copy(WB_D, WB_D_Temp2);
      Hadamard(WB_D_Temp2, WB_D_Temp, WB_D);

      //    WBL2NormSum = 0.0;
      Axpy((DataType)-lr, WB_D, WB);
    }

    float get_learning_rate() const { return lr; }

    void set_learning_rate(float _lr) { lr = _lr; }

    bool saveToCheckpoint(int fd, const char* filename, uint64_t* bytes) {
      //    writeDist(fd, filename, WB_D_Cache, bytes);
      return true;
    }

    bool loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes) {
      //    readDist(fd, filename, WB_D_Cache, bytes);
      return true;
    }

    bool saveToCheckpointShared(persist& p, int Index) {
      char name[512];
    
      // current learning rate value
      if (p.m_rank == 0) {
        sprintf(name, "L%d_learning_rate", Index);
        p.write_float(persist_type::train, name, lr);
      }
    
      // build the name of the checkpoint file
      sprintf(name, "L%d_adagrad_%dx%d", Index, WB_D_Cache.Height(), WB_D_Cache.Width());
      p.write_distmat(persist_type::train, name, (DistMat*)&WB_D_Cache);
    
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
    
      // build the name of the checkpoint file
      sprintf(name, "L%d_adagrad_%dx%d.bin", Index, WB_D_Cache.Height(), WB_D_Cache.Width());
      p.read_distmat(persist_type::train, name, (DistMat*)&WB_D_Cache);
    
      return true;
    }

  };

  class Adagrad_factory : public Optimizer_factory {
  public:
    Adagrad_factory(lbann_comm* comm, float lr=0.01, float epsilon=1e-6);
    ~Adagrad_factory();
    Optimizer *create_optimizer(matrix_format format=matrix_format::MC_MR);
    const string name() { return "adagrad"; }

  public:
    lbann_comm* comm;
    float lr;
    float epsilon;
  };

}

#endif // LBANN_OPTIMIZER_ADAGRAD_HPP
