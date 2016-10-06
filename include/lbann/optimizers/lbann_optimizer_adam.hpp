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
// lbann_optimizer_adam .hpp .cpp - SGD with Adam
// Reference:
// Kingma, D. and Ba, J. 2014. Adam: A Method for Stochastic Optimization.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_ADAM_HPP
#define LBANN_OPTIMIZER_ADAM_HPP

#include "lbann/optimizers/lbann_optimizer.hpp"

namespace lbann {

template <class _DistMat>
class Adam : public Optimizer {
public:
  /**
   * Constructor.
   * @param lr Learning rate (step size).
   * @param rho1 Exponential decay rate of the first moment estimate.
   * @param rho2 Exponential decay rate of the second moment estimate.
   * @param eps Small constant for numerical stability.
   */
  Adam(lbann_comm* comm, float lr, float rho1 = 0.9f, float rho2 = 0.999f,
       float eps = 1e-8f) :
    comm(comm), lr(lr), rho1(rho1), rho2(rho2), eps(eps),
    cur_rho1(1.0f), cur_rho2(1.0f),
    moment1_hist(comm->get_model_grid()),
    moment2_hist(comm->get_model_grid()) {
    if (comm->am_model_master()) {
      std::cout << "Initializing Adam optimizer with lr=" << lr <<
        " eps=" << eps << " rho1=" << rho1 << " rho2" << rho2 << std::endl;
    }
  }
  ~Adam() {
    moment1_hist.Empty();
    moment2_hist.Empty();
  }

  void setup(int input_dim, int num_neurons) {
    Zeros(moment1_hist, num_neurons, input_dim);
    Zeros(moment2_hist, num_neurons, input_dim);
  }

  void update_weight_bias_matrix(ElMat& WB_D, ElMat& WB) {
    // Update exponential decay rates.
    cur_rho1 *= rho1;
    cur_rho2 *= rho2;
    // Compute the correction factor.
    const float correction = std::sqrt(1.0f - cur_rho2) / (1.0f - cur_rho1);
    // Update the biased first and second moments.
    Scale(rho1, moment1_hist);
    Axpy(1.0f - rho1, WB_D, moment1_hist);
    Scale(rho2, moment2_hist);
    EntrywiseMap(WB_D,
                 std::function<DataType(DataType)>(
                   [] (DataType x) { return x * x; }));
    Axpy(1.0f - rho2, WB_D, moment2_hist);
    // Compute the update. Use WB_D as a temporary as it's no longer needed.
    Copy(moment2_hist, WB_D);
    EntrywiseMap(WB_D,
                 std::function<DataType(DataType)>(
                   [this] (DataType x) { return 1.0f / std::sqrt(x + eps); }));
    Hadamard(moment1_hist, WB_D, WB_D);
    Axpy(-lr * correction, WB_D, WB);
  }

  float get_learning_rate() const { return lr; }

  void set_learning_rate(float _lr) { lr = _lr; }

private:
  lbann_comm* comm;
  /** Learning rate. */
  float lr;
  /** Exponential decay rate for first moment estimate. */
  float rho1;
  /** Exponential decay rate for second moment estimate. */
  float rho2;
  /** Numerical stabilizer. */
  float eps;
  /** History of the first moment. */
  _DistMat moment1_hist;
  /** History of the second moment. */
  _DistMat moment2_hist;
  /** Current decay for the first moment. */
  float cur_rho1;
  /** Current decay for the second moment. */
  float cur_rho2;
};

class Adam_factory : public Optimizer_factory {
public:
  Adam_factory(lbann_comm* comm, float lr, float rho1 = 0.9f,
               float rho2 = 0.999f, float eps = 1e-8f);
  ~Adam_factory();
  Optimizer* create_optimizer(matrix_format format = matrix_format::MC_MR);
private:
  lbann_comm* comm;
  float lr;
  float rho1;
  float rho2;
  float eps;
};

}  // namespace lbann

#endif  // LBANN_OPTIMIZER_ADAM_HPP
