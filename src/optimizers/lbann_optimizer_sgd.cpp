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

#include "lbann/optimizers/lbann_optimizer_sgd.hpp"
#include "lbann/utils/lbann_exception.hpp"

using namespace std;
using namespace El;

lbann::sgd::sgd
(lbann_comm *comm,
 DataType learning_rate,
 DataType momentum,
 DataType decay,
 bool nesterov)
  : optimizer(comm, "sgd", learning_rate),
    m_momentum(momentum),
    m_decay(decay),
    m_nesterov(nesterov) {}

lbann::sgd::~sgd() {
  if(m_velocity) {
    delete m_velocity;
  }
}

void lbann::sgd::setup(AbsDistMat *parameters) {
  optimizer::setup(parameters);

  // Initialize iteration count
  m_iterations = 0;

  // Initialize velocity matrix
  if(m_momentum != DataType(0)) {
    switch(m_matrix_format) {
    case matrix_format::MC_MR:
      m_velocity = new DistMat(comm->get_model_grid());
      break;
    case matrix_format::STAR_STAR:
      m_velocity = new StarMat(comm->get_model_grid());
      break;
    case matrix_format::MC_STAR:
      m_velocity = new RowSumMat(comm->get_model_grid());
      break;
    case matrix_format::STAR_VC:
      m_velocity = new StarVCMat(comm->get_model_grid());
      break;
    default:
      throw lbann_exception("lbann_optimizer_sgd: invalid data layout");
    }
    Zeros(*m_velocity, m_height, m_width);
  }

}

void lbann::sgd::update(const AbsDistMat *gradient) {

  // Update learning rate and iteration count
  m_learning_rate *= DataType(1) / ( 1 + m_decay * m_iterations );
  ++m_iterations;

  if(m_momentum == DataType(0)) {
    // Vanilla SGD
    Axpy(-m_learning_rate, *gradient, *m_parameters);
  } else {

    // Get local matrix data
    const Int local_height = m_parameters->LocalHeight();
    const Int local_width = m_parameters->LocalWidth();
    DataType *parameters_buffer = m_parameters->Buffer();
    const Int parameters_ldim = m_parameters->LDim();
    const DataType *gradient_buffer = gradient->LockedBuffer();
    const Int gradient_ldim = gradient->LDim();
    DataType *velocity_buffer = m_velocity->Buffer();
    const Int velocity_ldim = m_velocity->LDim();

    // Check if matrix data is contiguous
    if(parameters_ldim != local_height
        || gradient_ldim != local_height
        || velocity_ldim != local_height) {
      // (Nesterov) momentum SGD for non-contiguous data
      #pragma omp parallel for collapse(2)
      for(Int j=0; j<local_width; ++j) {
        for(Int i=0; i<local_height; ++i) {
          const DataType g = gradient_buffer[i+j*gradient_ldim];
          DataType& v = velocity_buffer[i+j*velocity_ldim];
          DataType& x = parameters_buffer[i+j*parameters_ldim];
          v = m_momentum * v - m_learning_rate * g;
          x += m_nesterov ? m_momentum * v - m_learning_rate * g : v;
        }
      }
    } else {
      if(m_nesterov) {
        // Nesterov's accelerated gradient descent for contiguous data
        #pragma omp parallel for
        for(Int i=0; i<local_height*local_width; ++i) {
          DataType& x = parameters_buffer[i];
          const DataType g = gradient_buffer[i];
          DataType& v = velocity_buffer[i];
          v = m_momentum * v - m_learning_rate * g;
          x += m_momentum * v - m_learning_rate * g;
        }
      } else {
        // Momentum SGD for contiguous data
        #pragma omp parallel for
        for(Int i=0; i<local_height*local_width; ++i) {
          DataType& x = parameters_buffer[i];
          const DataType g = gradient_buffer[i];
          DataType& v = velocity_buffer[i];
          v = m_momentum * v - m_learning_rate * g;
          x += v;
        }
      }
    }

  }

}

#if 0
bool saveToCheckpoint(int fd, const char *filename, uint64_t *bytes) {
  //    writeDist(fd, filename, velocity, bytes);
  return true;
}

bool loadFromCheckpoint(int fd, const char *filename, uint64_t *bytes) {
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
  p.write_distmat(persist_type::train, name, (DistMat *)&velocity);

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
  p.read_distmat(persist_type::train, name, (DistMat *)&velocity);

  return true;
}
#endif

lbann::sgd_factory::sgd_factory
(lbann_comm *comm,
 DataType learning_rate,
 DataType momentum,
 DataType decay,
 bool nesterov)
  : optimizer_factory(comm, "sgd"),
    m_learning_rate(learning_rate),
    m_momentum(momentum),
    m_decay(decay),
    m_nesterov(nesterov) {}

lbann::sgd_factory::~sgd_factory() {}

lbann::optimizer *lbann::sgd_factory::create_optimizer() {
  return new sgd(comm, m_learning_rate, m_momentum, m_decay, m_nesterov);
}
