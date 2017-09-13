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
// learning_regularizer.hpp - Parent class for regularizers that learn
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LEARNING_REGULARIZER_HPP_INCLUDED
#define LBANN_LEARNING_REGULARIZER_HPP_INCLUDED

#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/layers/optimizable_layer.hpp"

namespace lbann {

/**
 * Regularizers with parameters and optimizer.
 */
class learning_regularizer : public regularizer_layer, public optimizable_layer {
 protected:
  optimizer *m_optimizer;
 public:
  learning_regularizer(int index, lbann_comm *comm, optimizer *opt) :
    regularizer_layer(index, comm), m_optimizer(opt) {}
  learning_regularizer(const learning_regularizer& other) :
    regularizer_layer(other) {
    m_optimizer = other.m_optimizer->copy();
  }
  learning_regularizer& operator=(const learning_regularizer& other) {
    regularizer_layer::operator=(other);
    if (m_optimizer) {
      delete m_optimizer;
    }
    if (other.m_optimizer) {
      m_optimizer = other.m_optimizer->copy();
    }
    return *this;
  }
  ~learning_regularizer() {
    delete m_optimizer;
  }

  template<data_layout T_layout> inline void initialize_distributed_matrices() {
    regularizer_layer::initialize_distributed_matrices<T_layout>();
  }

  optimizer *get_optimizer() const override {
    return m_optimizer;
  }
};

}  // namespace lbann

#endif // LBANN_LEARNING_REGULARIZER_HPP_INCLUDED
