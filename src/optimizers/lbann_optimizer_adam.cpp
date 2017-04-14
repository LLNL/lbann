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
  const DataType correction = ( Sqrt(DataType(1) - m_current_beta2)
                                / (DataType(1) - m_current_beta1) );
  
  // Get local matrix data
  const Int local_height = m_parameters->LocalHeight();
  const Int local_width = m_parameters->LocalWidth();
  DataType* parameter_buffer = m_parameters->Buffer();
  const DataType* gradient_buffer = gradient->LockedBuffer();
  DataType* moment1_buffer = m_moment1->Buffer();
  DataType* moment2_buffer = m_moment2->Buffer();
  
  // Update parameters
  // Note: we assume data is contiguous
#pragma omp parallel for
  for(Int i=0; i<local_height*local_width; ++i) {
    const DataType g = gradient_buffer[i];
    moment1_buffer[i] = ( m_beta1 * moment1_buffer[i] 
                          + (DataType(1) - m_beta1) * g );
    moment2_buffer[i] = ( m_beta2 * moment2_buffer[i] 
                          + (DataType(1) - m_beta1) * g * g );
    parameter_buffer[i] -= ( m_learning_rate * correction * moment1_buffer[i]
                             / (Sqrt(moment2_buffer[i]) + m_eps) );
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
