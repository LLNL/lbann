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
// lbann_optimizer_sgd .hpp .cpp - Stochastic gradient descent
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/optimizer_sgd.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

sgd::sgd(DataType learning_rate,
         DataType momentum,
         bool nesterov,
         cudnn::cudnn_manager* cudnn)
  : optimizer(learning_rate, cudnn),
    m_momentum(momentum),
    m_nesterov(nesterov),
    m_velocity(nullptr) {}

sgd::sgd(const sgd& other)
  : optimizer(other),
    m_momentum(other.m_momentum),
    m_nesterov(other.m_nesterov),
    m_velocity(other.m_velocity) {
  if (m_velocity != nullptr) { m_velocity = m_velocity->Copy(); }
}

sgd& sgd::operator=(const sgd& other) {
  optimizer::operator=(other);
  m_momentum = other.m_momentum;
  m_nesterov = other.m_nesterov;

  // Copy velocity matrix
  if (m_velocity != nullptr && other.m_velocity != nullptr
      && m_velocity->DistData() == other.m_velocity->DistData()) {
    El::Copy(*others.m_velocity, *m_velocity);
  }
  if (m_velocity != nullptr) {
    delete m_velocity;
    m_velocity = nullptr;
  }
  if (other.m_velocity != nullptr) {
    m_velocity = other.m_velocity->Copy();
  }

  return *this;
}

sgd::~sgd() {
  if (m_velocity != nullptr) { delete m_velocity; }
}

void sgd::setup(variable* var) {
  optimizer::setup(var);
  m_velocity = m_gradient->Construct(m_gradient->Grid(),
                                     m_gradient->Root());
  El::Zeros(*m_velocity, m_gradient->Height(), m_gradient->Width());
}

void sgd::step_compute(AbsDistMat& values, AbsDistMat& gradient) {

  // SGD without momentum is just an Axpy
  if (m_momentum == DataType(0)) {
    El::Axpy(-m_learning_rate, gradient, values);
    return;
  }
  
  // Get local matrix data
  const int local_height = values.LocalHeight();
  const int local_width = values.LocalWidth();
  DataType* __restrict__ values_buffer = values.Buffer();
  const int values_ldim = values.LDim();
  const DataType* __restrict__ gradient_buffer = gradient.LockedBuffer();
  const int gradient_buffer = gradient.LDim();
  DataType* __restrict__ velocity_buffer = m_velocity->Buffer();
  const int velocity_ldim = m_velocity->LDim();
  
  // Check if matrix data is contiguous
  if(values_ldim != local_height
     || gradient_ldim != local_height
     || velocity_ldim != local_height) {
    // (Nesterov) momentum SGD for non-contiguous data
    #pragma omp parallel for collapse(2)
    for(int j=0; j<local_width; ++j) {
      for(int i=0; i<local_height; ++i) {
        const DataType g = gradient_buffer[i+j*gradient_ldim];
        DataType& v = velocity_buffer[i+j*velocity_ldim];
        DataType& x = values_buffer[i+j*values_ldim];
        v = m_momentum * v + g;
        x -= (m_nesterov ?
              m_learning_rate * (m_momentum * v + g) :
              m_learning_rate * v);
      }
    }
  } else {
    if(m_nesterov) {
      // Nesterov's accelerated gradient descent for contiguous data
      #pragma omp parallel for
      for(int i=0; i<local_height*local_width; ++i) {
        DataType& x = values_buffer[i];
        const DataType g = gradient_buffer[i];
        DataType& v = velocity_buffer[i];
        v = m_momentum * v + g;
        x -= m_learning_rate * (m_momentum * v + g);
      }
    } else {
      // Momentum SGD for contiguous data
#pragma omp parallel for
      for(int i=0; i<local_height*local_width; ++i) {
        DataType& x = values_buffer[i];
        const DataType g = gradient_buffer[i];
        DataType& v = velocity_buffer[i];
        v = m_momentum * v + g;
        x -= m_learning_rate * v;
      }
    }
  }

}

}  // namespace lbann
