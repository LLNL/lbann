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

#include "lbann/optimizers/lbann_optimizer_adam.hpp"
#include "lbann/utils/lbann_exception.hpp"

lbann::adam::adam
(lbann_comm* comm,
 DataType learning_rate,
 DataType beta1,
 DataType beta2,
 DataType eps)
  : optimizer(comm, "adam", learning_rate),
    m_beta1(beta1),
    m_beta2(beta2),
    m_eps(eps),
    m_current_beta1(1),
    m_current_beta2(1) {}

lbann::adam::~adam()
{
  if(m_moment1)
    delete m_moment1;
  if(m_moment2)
    delete m_moment2;
}

void lbann::adam::setup(AbsDistMat* parameters)
{
  optimizer::setup(parameters);

  // Initialize Adam cache
  switch(m_matrix_format) {
  case matrix_format::MC_MR:
    m_moment1 = new DistMat(comm->get_model_grid());
    m_moment2 = new DistMat(comm->get_model_grid());
    break;
  case matrix_format::STAR_STAR:
    m_moment1 = new StarMat(comm->get_model_grid());
    m_moment2 = new StarMat(comm->get_model_grid());
    break;
  case matrix_format::MC_STAR:
    m_moment1 = new RowSumMat(comm->get_model_grid());
    m_moment2 = new RowSumMat(comm->get_model_grid());
    break;
  case matrix_format::STAR_VC:
    m_moment1 = new StarVCMat(comm->get_model_grid());
    m_moment2 = new StarVCMat(comm->get_model_grid());
    break;
  default:
    throw lbann_exception("lbann_optimizer_adam: invalid data layout");
  }
  Zeros(*m_moment1, m_height, m_width);
  Zeros(*m_moment2, m_height, m_width);

}

void lbann::adam::update(const AbsDistMat* gradient)
{

  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  // Precompute the bias correction and learning rate.
  const DataType correction = m_learning_rate *
    (Sqrt(DataType(1) - m_current_beta2)
     / (DataType(1) - m_current_beta1));
  
  // Get local matrix data
  const Int local_height = m_parameters->LocalHeight();
  const Int local_width = m_parameters->LocalWidth();
  DataType* parameters_buffer = m_parameters->Buffer();
  const Int parameters_ldim = m_parameters->LDim();
  const DataType* gradient_buffer = gradient->LockedBuffer();
  const Int gradient_ldim = gradient->LDim();
  DataType* moment1_buffer = m_moment1->Buffer();
  const Int moment1_ldim = m_moment1->LDim();
  DataType* moment2_buffer = m_moment2->Buffer();
  const Int moment2_ldim = m_moment2->LDim();

  // Check if matrix data is contiguous
  if(parameters_ldim != local_height
     || gradient_ldim != local_height
     || moment1_ldim != local_height
     || moment2_ldim != local_height) {
    // Update with non-contiguous data
#pragma omp parallel for collapse(2)
    for(Int j=0; j<local_width; ++j) {
      for(Int i=0; i<local_height; ++i) {
        DataType& x = parameters_buffer[i+j*parameters_ldim];
        const DataType g = gradient_buffer[i+j*gradient_ldim];
        DataType& m1 = moment1_buffer[i+j*moment1_ldim];
        DataType& m2 = moment2_buffer[i+j*moment2_ldim];
        m1 = m_beta1 * m1 + (DataType(1) - m_beta1) * g;
        m2 = m_beta2 * m2 + (DataType(1) - m_beta2) * g * g;
        x -= correction * m1 / (Sqrt(m2) + m_eps);
      }
    }
  }
  else {
    // Update with contiguous data
#pragma omp parallel for
    for(Int i=0; i<local_height*local_width; ++i) {
      DataType& x = parameters_buffer[i];
      // We add eps here because sometimes the gradient is small enough that
      // g*g can become denormalized, which can significantly impact the
      // performance.
      const DataType g = gradient_buffer[i] + m_eps;
      DataType& m1 = moment1_buffer[i];
      DataType& m2 = moment2_buffer[i];
      m1 = m_beta1 * m1 + (DataType(1) - m_beta1) * g;
      m2 = m_beta2 * m2 + (DataType(1) - m_beta2) * g * g;
      x -= correction * m1 / (Sqrt(m2) + m_eps);
    }
  }

}

#if 0
  bool saveToCheckpointShared(persist& p, int Index) {
    char name[512];
  
    // current learning rate value
    if (p.m_rank == 0) {
      sprintf(name, "L%d_learning_rate", Index);
      p.write_float(persist_type::train, name, lr);
    }
  
    // current rho1 value
    if (p.m_rank == 0) {
      sprintf(name, "L%d_cur_rho1", Index);
      p.write_float(persist_type::train, name, cur_rho1);
    }
  
    // current rho2 value
    if (p.m_rank == 0) {
      sprintf(name, "L%d_cur_rho2", Index);
      p.write_float(persist_type::train, name, cur_rho2);
    }
  
    // checkpoint matrix for first moment
    sprintf(name, "L%d_adam_moment1_%dx%d",
      Index, moment1_hist.Height(), moment1_hist.Width());
    bool rc1 = p.write_distmat(persist_type::train, name, (DistMat*)&moment1_hist);
  
    // checkpoint matrix for second moment
    sprintf(name, "L%d_adam_moment2_%dx%d",
      Index, moment2_hist.Height(), moment2_hist.Width());
    bool rc2 = p.write_distmat(persist_type::train, name, (DistMat*)&moment2_hist);
  
    return (rc1 && rc2);
  }
  
  bool loadFromCheckpointShared(persist& p, int Index) {
    char name[512];
  
    // current learning rate value
    if (p.m_rank == 0) {
      sprintf(name, "L%d_learning_rate", Index);
      p.read_float(persist_type::train, name, &lr);
    }
    MPI_Bcast(&lr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
    // current rho1 value
    if (p.m_rank == 0) {
      sprintf(name, "L%d_cur_rho1", Index);
      p.read_float(persist_type::train, name, &cur_rho1);
    }
    MPI_Bcast(&cur_rho1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
    // current rho2 value
    if (p.m_rank == 0) {
      sprintf(name, "L%d_cur_rho2", Index);
      p.read_float(persist_type::train, name, &cur_rho2);
    }
    MPI_Bcast(&cur_rho2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
    // checkpoint matrix for first moment
    sprintf(name, "L%d_adam_moment1_%dx%d.bin",
      Index, moment1_hist.Height(), moment1_hist.Width());
    bool rc1 = p.read_distmat(persist_type::train, name, (DistMat*)&moment1_hist);
  
    // checkpoint matrix for second moment
    sprintf(name, "L%d_adam_moment2_%dx%d.bin",
      Index, moment2_hist.Height(), moment2_hist.Width());
    bool rc2 = p.read_distmat(persist_type::train, name, (DistMat*)&moment2_hist);
  
    return (rc1 && rc2);
  }
#endif

lbann::adam_factory::adam_factory
(lbann_comm* comm,
 DataType learning_rate,
 DataType beta1,
 DataType beta2,
 DataType eps)
  : optimizer_factory(comm, "adam"),
    m_learning_rate(learning_rate),
    m_beta1(beta1),
    m_beta2(beta2),
    m_eps(eps) {}

lbann::adam_factory::~adam_factory() {}

lbann::optimizer* lbann::adam_factory::create_optimizer()
{
  return new adam(comm, m_learning_rate, m_beta1, m_beta2, m_eps);
}
