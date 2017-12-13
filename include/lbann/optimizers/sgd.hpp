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
// sgd .hpp .cpp - Stochastic gradient descent optimizer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_SGD_HPP
#define LBANN_OPTIMIZER_SGD_HPP

#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

/** Stochastic gradient descent optimizer.
 *  Supports momentum and Nesterov acceleration.
 */
class sgd : public optimizer {

 public:

  /** Constructor. */
  sgd(lbann_comm *comm,
      DataType learning_rate,
      DataType momentum = DataType(0),
      bool nesterov = false);

  /** Copy constructor. */
  sgd(const sgd& other);
  /** Copy assignment operator. */
  sgd& operator=(const sgd& other);
  /** Destructor. */
  ~sgd() override;
  /** Create a copy. */
  sgd* copy() const override { return new sgd(*this); }

  /** Get the optimizer name. */
  std::string get_type() const override { return "sgd"; }
  /** Get a human-readable description of the optimizer. */
  std::string get_description() const override;

  /** Setup optimizer. */
  void setup(weights& w) override;

  /** Perform the computation in an optimization step. */
  void step_compute(AbsDistMat& values, const AbsDistMat& gradient) override;
#ifdef __LIB_CUDNN
  /** Perform the computation in an optimization step on GPU. */
  void step_compute_gpu(std::vector<DataType*> values_d,
                        std::vector<DataType*> gradient_d) override;
#endif // __LIB_CUDNN
  bool save_to_checkpoint_shared(persist& p, std::string m_name) override {
    optimizer::save_to_checkpoint_shared(p, m_name);
    char l_name[512];
    if (p.get_rank() == 0) {
      sprintf(l_name, "%s_learning_rate", m_name.c_str());
      p.write_float(persist_type::train, l_name, m_learning_rate);

      sprintf(l_name, "%s_momentum", m_name.c_str());
      p.write_float(persist_type::train, l_name, m_momentum);

    }
    //sprintf(l_name, "gradient_sgd_%s", m_name.c_str());
    //p.write_distmat(persist_type::train, l_name, (DistMat *)m_gradient);
    sprintf(l_name, "%s_velocity_sgd", m_name.c_str());
    p.write_distmat(persist_type::train, l_name, (DistMat *)m_velocity);
    return true;

  }

  bool load_from_checkpoint_shared(persist& p, std::string m_name) override  {
    //optimizer::load_from_checkpoint_shared(p, m_name);
    char l_name[512];
    if (p.get_rank() == 0) {
      sprintf(l_name, "%s_learning_rate", m_name.c_str());
      p.read_float(persist_type::train, l_name, &m_learning_rate);

      sprintf(l_name, "%s_momentum", m_name.c_str());
      p.read_float(persist_type::train, l_name, &m_momentum);

    }
    MPI_Bcast(&m_learning_rate, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m_momentum, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //sprintf(l_name, "gradient_sgd_%s.bin", m_name.c_str());
    //p.read_distmat(persist_type::train, l_name, (DistMat *)m_gradient);
    sprintf(l_name, "%s_velocity_sgd.bin", m_name.c_str());
    p.read_distmat(persist_type::train, l_name, (DistMat *)m_velocity);
    return true;
   }
 private:

  /** Momentum. */
  DataType m_momentum;
  /** Nesterov acceleration. */
  bool m_nesterov;
  /** Velocity term for momentum SGD. */
  AbsDistMat* m_velocity;

#ifdef __LIB_CUDNN
  /** GPU memory for velocity. */
  std::vector<DataType*> m_velocity_d;
#endif // __LIB_CUDNN

};

} // namespace lbann

#endif // LBANN_OPTIMIZER_SGD_HPP
