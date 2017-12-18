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

#include "lbann/optimizers/sgd.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

sgd::sgd(lbann_comm *comm,
         DataType learning_rate,
         DataType momentum,
         bool nesterov)
  : optimizer(comm, learning_rate),
    m_momentum(momentum),
    m_nesterov(nesterov),
    m_velocity(nullptr) {}

sgd::sgd(const sgd& other)
  : optimizer(other),
    m_momentum(other.m_momentum),
    m_nesterov(other.m_nesterov),
    m_velocity(other.m_velocity) {
  if (m_velocity != nullptr) { m_velocity = m_velocity->Copy(); }
  #ifdef LBANN_HAS_CUDNN
  if (m_cudnn != nullptr && other.m_weights != nullptr) {
    const int height = other.m_weights->get_height();
    const int width = other.m_weights->get_width();
    m_velocity_d = m_cudnn->copy(other.m_velocity_d, height, width);
  }
  #endif // LBANN_HAS_CUDNN
}

sgd& sgd::operator=(const sgd& other) {
  optimizer::operator=(other);
  m_momentum = other.m_momentum;
  m_nesterov = other.m_nesterov;

  // Copy velocity matrix
  if (m_velocity != nullptr && other.m_velocity != nullptr
      && m_velocity->DistData() == other.m_velocity->DistData()) {
    El::Copy(*other.m_velocity, *m_velocity);
  }
  else {
    if (m_velocity != nullptr) { delete m_velocity; }
    m_velocity = other.m_velocity;
    if (m_velocity != nullptr) { m_velocity = m_velocity->Copy(); }
  }

  // Copy GPU data
  #ifdef LBANN_HAS_CUDNN
  if (m_cudnn != nullptr && other.m_weights != nullptr) {
    const int height = other.m_weights->get_height();
    const int width = other.m_weights->get_width();
    m_cudnn->deallocate_on_gpus(m_velocity_d);
    m_velocity_d = m_cudnn->copy(other.m_velocity_d, height, width);
  }
  #endif // LBANN_HAS_CUDNN

  return *this;
}

sgd::~sgd() {
  if (m_velocity != nullptr) { delete m_velocity; }
  #ifdef LBANN_HAS_CUDNN
  if (m_cudnn != nullptr) {
    m_cudnn->deallocate_on_gpus(m_velocity_d);
  }
  #endif // LBANN_HAS_CUDNN
}

std::string sgd::get_description() const {
  std::stringstream ss;
  ss << optimizer::get_description() << ", "
     << "momentum=" << m_momentum << ", "
     << "nesterov=" << m_nesterov;
  return ss.str();
}

void sgd::setup(weights& w) {
  optimizer::setup(w);

  // Allocate matrices
  const int height = m_gradient->Height();
  const int width = m_gradient->Width();
  m_velocity = m_gradient->Construct(m_gradient->Grid(),
                                     m_gradient->Root());
  El::Zeros(*m_velocity, height, width);

  // Allocate GPU objects
  if (m_cudnn != nullptr) {
#ifdef LBANN_HAS_CUDNN
    m_cudnn->allocate_on_gpus(m_velocity_d, height, width);
#endif // LBANN_HAS_CUDNN
  }
  
}

void sgd::step_compute(AbsDistMat& values, const AbsDistMat& gradient) {

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
  const int gradient_ldim = gradient.LDim();
  DataType* __restrict__ velocity_buffer = m_velocity->Buffer();
  const int velocity_ldim = m_velocity->LDim();
  
  // Check if matrix data is contiguous
  if (values_ldim != local_height
      || gradient_ldim != local_height
      || velocity_ldim != local_height) {
    // (Nesterov) momentum SGD for non-contiguous data
    #pragma omp parallel for collapse(2)
    for (int j=0; j<local_width; ++j) {
      for (int i=0; i<local_height; ++i) {
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
    if (m_nesterov) {
      // Nesterov's accelerated gradient descent for contiguous data
      #pragma omp parallel for
      for (int i=0; i<local_height*local_width; ++i) {
        DataType& x = values_buffer[i];
        const DataType g = gradient_buffer[i];
        DataType& v = velocity_buffer[i];
        v = m_momentum * v + g;
        x -= m_learning_rate * (m_momentum * v + g);
      }
    } else {
      // Momentum SGD for contiguous data
#pragma omp parallel for
      for (int i=0; i<local_height*local_width; ++i) {
        DataType& x = values_buffer[i];
        const DataType g = gradient_buffer[i];
        DataType& v = velocity_buffer[i];
        v = m_momentum * v + g;
        x -= m_learning_rate * v;
      }
    }
  }

}

////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

  bool sgd::save_to_checkpoint_shared(persist& p, std::string name_prefix) {
    optimizer::save_to_checkpoint_shared(p, name_prefix);
    
    if(p.get_rank() == 0){
      pack_scalars(p);
    }

    char l_name[512];
    sprintf(l_name, "%s_optimizer_velocity_%lldx%lld", name_prefix.c_str(), m_velocity->Height(), m_velocity->Width());
    p.write_distmat(persist_type::train, l_name, (DistMat *)m_velocity);
    
    return true;
  }

  bool sgd::load_from_checkpoint_shared(persist& p, std::string name_prefix) {
    optimizer::load_from_checkpoint_shared(p, name_prefix);
    struct packing_header header;
    if (p.get_rank() == 0) {
      unpack_scalars(p, &header);
    }
    
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    unpack_header(header);
    char l_name[512];
    
    sprintf(l_name, "%s_optimizer_velocity_%lldx%lld.bin", name_prefix.c_str(), m_velocity->Height(), m_velocity->Width());
    p.read_distmat(persist_type::train, l_name, (DistMat *)m_velocity);
    
    return true;
  }

}  // namespace lbann
