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
// adagrad .hpp .cpp - SGD with AdaGrad optimizer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_ADAGRAD_HPP
#define LBANN_OPTIMIZER_ADAGRAD_HPP

#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

/** AdaGrad optimizer. */
class adagrad : public optimizer {
 public:

  /** Constructor. */
  adagrad(lbann_comm *comm,
          DataType learning_rate,
          DataType eps = DataType(1e-8));

  /** Copy constructor. */
  adagrad(const adagrad& other);
  /** Copy assignment operator. */
  adagrad& operator=(const adagrad& other);
  /** Destructor. */
  ~adagrad() override;
  /** Create a copy. */
  adagrad* copy() const override { return new adagrad(*this); }

  /** Get the optimizer name. */
  std::string get_type() const override { return "adagrad"; }
  /** Get a human-readable description of the optimizer. */
  std::string get_description() const override;


  /** Setup optimizer. */
  void setup(weights& w) override;

  /** Perform the computation in an optimization step. */
  void step_compute(AbsDistMat& values, const AbsDistMat& gradient) override;

  /// Set parameters to optimize and initialize optimizer
  void setup(AbsDistMat *parameters) override;
  /// Update parameters using objective function gradient
  void update(const AbsDistMat *gradient) override;
  std::string name() const override { return "adagrad"; }

  bool saveToCheckpointShared(persist& p, std::string m_name) override {
    char l_name[512];
    if (p.get_rank() == 0) {
      sprintf(l_name, "%s_learning_rate", m_name.c_str());
      p.write_float(persist_type::train, l_name, m_learning_rate);
    }
    sprintf(l_name, "cache_adagrad_%s", m_name.c_str());
    p.write_distmat(persist_type::train, l_name, (DistMat *)m_cache);
    return true;

  }

  bool loadFromCheckpointShared(persist& p, std::string m_name) override {
    char l_name[512];
    if (p.get_rank() == 0) {
      sprintf(l_name, "%s_learning_rate", m_name.c_str());
      p.read_float(persist_type::train, l_name, &m_learning_rate);
    }
    MPI_Bcast(&m_learning_rate, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    sprintf(l_name, "cache_adagrad_%s.bin", m_name.c_str());
    p.read_distmat(persist_type::train, l_name, (DistMat *)m_cache);
    return true;

  }

 private:

  /** Small factor to avoid division by zero. */
  DataType m_eps;
  /** AdaGrad cache. */
  AbsDistMat *m_cache;

};

} // namespace lbann

#endif // LBANN_OPTIMIZER_ADAGRAD_HPP
