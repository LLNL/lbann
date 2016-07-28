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
// lbann_optimizer_rmsprop .hpp .cpp - Stochastic gradient descent optimizer with RMSprop
//
/// Inspired by Kera.io implementation
/// Stochastic gradient descent with RMSprop.
///  lr: float >= 0. Learning rate.
///  rho: float >= 0.
///  epsilon: float >= 0. Fuzz factor.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_RMSPROP_HPP
#define LBANN_OPTIMIZER_RMSPROP_HPP

#include "lbann/optimizers/lbann_optimizer.hpp"
#include <sys/stat.h>

namespace lbann
{
  template <class _DistMat>
  class RMSprop : public Optimizer {

  public:
    float LearnRate;
    //    float lr;
    float rho;
    float epsilon;

    lbann_comm* comm;
    _DistMat     WB_D_Cache;     // Cache of Weights and Bias Gradient (current time t - 1)
    _DistMat     WB_D_Temp;      // Temporary for Weights and Bias Gradient computation
    _DistMat     WB_D_Temp2;     // Temporary for Weights and Bias Gradient computation

  private:
    static inline DataType _sq(DataType x) { return (x * x); }
    static inline DataType _sqrt(DataType x) { return (1 / sqrt(x + 1e-8)); }

  public:
    RMSprop(lbann_comm* comm, float lr, float rho, float epsilon)
      : LearnRate(lr), rho(rho), epsilon(epsilon), comm(comm),
        WB_D_Cache(comm->get_model_grid()),
        WB_D_Temp(comm->get_model_grid()),
        WB_D_Temp2(comm->get_model_grid()) {
      if (comm->am_model_master()) {
        printf("Initializing RMSprop optimizer with lr=%f, rho=%f, and epsilon=%f\n", lr, rho, epsilon);
      }
    }

    ~RMSprop() {
      WB_D_Cache.Empty();
      WB_D_Temp.Empty();
      WB_D_Temp2.Empty();
    }

    void setup(int input_dim, int num_neurons) {
      if (comm->am_model_master()) {
        printf("Setting up RMSprop optimizer with cache size %d x %d\n", num_neurons, input_dim);
      }
      Zeros(WB_D_Cache, num_neurons, input_dim);
      Zeros(WB_D_Temp, num_neurons, input_dim);
      Zeros(WB_D_Temp2, num_neurons, input_dim);  
      if (comm->am_model_master()) {
        printf("Setting up RMSprop optimizer with WB_D_Cache size %d x %d\n", WB_D_Cache.Height(), WB_D_Cache.Width());  
      }
    }

    void update_weight_bias_matrix(ElMat &WB_D, ElMat& WB) {
      // update accumulator
      // KERAS: for p, g, a, c in zip(params, grads, accumulators, constraints):
      // KERAS: new_a = self.rho * a + (1 - self.rho) * K.square(g)
      Scale(rho /*DecayRate*/, WB_D_Cache);
      Copy(WB_D, WB_D_Temp);
      EntrywiseMap(WB_D_Temp, std::function<DataType(DataType)>(_sq));
      Scale(1 - rho /*DecayRate*/, WB_D_Temp);
      Axpy(1., WB_D_Temp, WB_D_Cache);

      // update parameters
      // KERAS: new_p = p - self.lr * g / K.sqrt(new_a + self.epsilon)
      Copy(WB_D_Cache, WB_D_Temp);
      EntrywiseMap(WB_D_Temp, std::function<DataType(DataType)>(_sqrt));
      Copy(WB_D, WB_D_Temp2);
      Hadamard(WB_D_Temp2, WB_D_Temp, WB_D);

      //    WBL2NormSum = 0.0;
      Axpy((DataType)-LearnRate, WB_D, WB);
    }

    float get_learning_rate() const { return LearnRate; }
    void set_learning_rate(float _lr) { LearnRate = _lr; }

    bool saveToCheckpoint(int fd, const char* filename, uint64_t* bytes) {
      //    writeDist(fd, filename, WB_D_Cache, bytes);
      return true;
    }

    bool loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes) {
      //    readDist(fd, filename, WB_D_Cache, bytes);
      return true;
    }

    bool saveToCheckpointShared(const char* dir, int Index, uint64_t* bytes) {
      int rank = WB_D_Cache.Grid().Rank();

      char path[512];
      sprintf(path, "%s/WB_D_CACHE_L%d_%03dx%03d", dir, Index, WB_D_Cache.Height()-1, WB_D_Cache.Width()-1);
      if(rank == 0) {
        cout << "Saving layer " << Index << " to file " << path << endl;
      }
      Write(WB_D_Cache, path, BINARY, "");
      //Write_MPI(WB_D_Cache, path, BINARY, "");

      if (rank == 0) {
        *bytes += 2 * sizeof(int) + WB_D_Cache.Height() * WB_D_Cache.Width() * sizeof(DataType);
      }

      return true;
    }

    bool loadFromCheckpointShared(const char* dir, int Index, uint64_t* bytes) {
      int rank = WB_D_Cache.Grid().Rank();

      char path[512];
      struct stat buffer;

      // read in the cache of gradients for WB
      sprintf(path, "%s/WB_D_CACHE_L%d_%03dx%03d.bin", dir, Index, WB_D_Cache.Height()-1, WB_D_Cache.Width()-1);

      // check whether file exists
      int exists = 0;
      if (rank == 0 && stat(path, &buffer) == 0) {
        exists = 1;
      }
      MPI_Bcast(&exists, 1, MPI_INT, 0, MPI_COMM_WORLD);

      // read WB_D_Cache file
      if (rank == 0) {
        cout << "Restoring layer " << Index << " from file " << path << endl;
      }
      Read(WB_D_Cache, path, BINARY, 1);
      //Read_MPI(WB_D_Cache, path, BINARY, 1);

      if (rank == 0) {
        *bytes += 2 * sizeof(int) + WB_D_Cache.Height() * WB_D_Cache.Width() * sizeof(DataType);
      }
      return true;
    }
    
  };

  class RMSprop_factory : public Optimizer_factory {
  public:
    // Default values from Keras - it is recommended that they are left at their default values
    RMSprop_factory(lbann_comm* comm, float lr=0.001, float rho=0.9, float epsilon=1e-6);
    ~RMSprop_factory();
    Optimizer *create_optimizer(matrix_format format=matrix_format::MC_MR);

  public:
    //    float LearnRate;
    lbann_comm* comm;
    float lr;
    float rho;
    float epsilon;
  };

}

#endif // LBANN_OPTIMIZER_RMSPROP_HPP
