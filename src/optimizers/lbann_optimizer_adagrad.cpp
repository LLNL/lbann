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
// lbann_optimizer_adagrad .hpp .cpp - SGD with AdaGrad
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/utils/lbann_exception.hpp"

using namespace std;
using namespace El;

lbann::adagrad::adagrad
(lbann_comm *comm,
 DataType learning_rate,
 DataType eps)
  : optimizer(comm, "adagrad", learning_rate), m_eps(eps) {}

lbann::adagrad::~adagrad() {
  if(m_cache) {
    delete m_cache;
  }
}

void lbann::adagrad::setup(AbsDistMat *parameters) {
  optimizer::setup(parameters);

  // Initialize AdaGrad cache
  switch(m_matrix_format) {
  case matrix_format::MC_MR:
    m_cache = new DistMat(m_comm->get_model_grid());
    break;
  case matrix_format::STAR_STAR:
    m_cache = new StarMat(m_comm->get_model_grid());
    break;
  case matrix_format::MC_STAR:
    m_cache = new RowSumMat(m_comm->get_model_grid());
    break;
  case matrix_format::STAR_VC:
    m_cache = new StarVCMat(m_comm->get_model_grid());
    break;
  default:
    throw lbann_exception("lbann_optimizer_adagrad: invalid data layout");
  }
  Zeros(*m_cache, m_height, m_width);

}

void lbann::adagrad::update(const AbsDistMat *gradient) {

  // Get local matrix data
  const int local_height = m_parameters->LocalHeight();
  const int local_width = m_parameters->LocalWidth();
  DataType *parameters_buffer = m_parameters->Buffer();
  const int parameters_ldim = m_parameters->LDim();
  const DataType *gradient_buffer = gradient->LockedBuffer();
  const int gradient_ldim = gradient->LDim();
  DataType *cache_buffer = m_cache->Buffer();
  const int cache_ldim = m_cache->LDim();

  // Check if matrix data is contiguous
  if(parameters_ldim != local_height
      || gradient_ldim != local_height
      || cache_ldim != local_height) {
    // Update with non-contiguous data
    #pragma omp parallel for collapse(2)
    for(int j=0; j<local_width; ++j) {
      for(int i=0; i<local_height; ++i) {
        DataType& x = parameters_buffer[i+j*parameters_ldim];
        const DataType g = gradient_buffer[i+j*gradient_ldim];
        DataType& c = cache_buffer[i+j*cache_ldim];
        c += g * g;
        x -= m_learning_rate * g / (Sqrt(c) + m_eps);
      }
    }
  } else {
    // Update with contiguous data
    #pragma omp parallel for
    for(int i=0; i<local_height*local_width; ++i) {
      DataType& x = parameters_buffer[i];
      const DataType g = gradient_buffer[i];
      DataType& c = cache_buffer[i];
      c += g * g;
      x -= m_learning_rate * g / (Sqrt(c) + m_eps);
    }
  }

}

lbann::adagrad_factory::adagrad_factory
(lbann_comm *comm,
 DataType learning_rate,
 DataType eps)
  : optimizer_factory(comm, "adagrad"),
    m_eps(eps),
    m_learning_rate(learning_rate)
{}

lbann::adagrad_factory::~adagrad_factory() {}

lbann::optimizer *lbann::adagrad_factory::create_optimizer() {
  return new adagrad(m_comm, m_learning_rate, m_eps);
}
