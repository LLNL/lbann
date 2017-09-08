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

#include "lbann/optimizers/optimizer_adam.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

adam::adam(lbann_comm *comm, DataType learning_rate, DataType beta1,
           DataType beta2, DataType eps, cudnn::cudnn_manager *cudnn)
    : optimizer(comm, learning_rate, cudnn),
    m_beta1(beta1),
    m_beta2(beta2),
    m_eps(eps),
    m_current_beta1(1),
    m_current_beta2(1),
    m_moment1(nullptr),
    m_moment2(nullptr) {}

adam::adam(const adam& other)
  : optimizer(other), m_beta1(other.m_beta1), m_beta2(other.m_beta2),
    m_eps(other.m_eps), m_current_beta1(other.m_current_beta1),
    m_current_beta2(other.m_current_beta2), m_moment1(nullptr),
    m_moment2(nullptr) {
  if (other.m_moment1) {
    m_moment1 = other.m_moment1->Copy();
    m_moment2 = other.m_moment2->Copy();
    if (other.m_moment1_d.size() > 0) {
#ifdef __LIB_CUDNN
      int local_height = m_parameters->LocalHeight();
      int local_width = m_parameters->LocalWidth();
      m_cudnn->allocate_on_gpus(m_moment1_d, local_height, local_width);
      m_cudnn->copy_on_gpus(m_moment1_d, other.m_moment1_d, local_height, local_width);
      m_cudnn->allocate_on_gpus(m_moment2_d, local_height, local_width);
      m_cudnn->copy_on_gpus(m_moment2_d, other.m_moment2_d, local_height, local_width);
#endif
    }
  }
}

adam& adam::operator=(const adam& other) {
  optimizer::operator=(other);
  m_beta1 = other.m_beta1;
  m_beta2 = other.m_beta2;
  m_eps = other.m_eps;
  m_current_beta1 = other.m_current_beta1;
  m_current_beta2 = other.m_current_beta2;
  if (m_moment1) {
    delete m_moment1;
    delete m_moment2;
  }
  if (m_moment1_d.size() > 0) {
#ifdef __LIB_CUDNN
    m_cudnn->deallocate_on_gpus(m_moment1_d);
    m_cudnn->deallocate_on_gpus(m_moment2_d);
#endif
    m_moment1_d.clear();
    m_moment2_d.clear();
  }
  if (other.m_moment1) {
    m_moment1 = other.m_moment1->Copy();
    m_moment2 = other.m_moment2->Copy();
    if (other.m_moment1_d.size() > 0) {
#ifdef __LIB_CUDNN
      int local_height = m_parameters->LocalHeight();
      int local_width = m_parameters->LocalWidth();
      m_cudnn->allocate_on_gpus(m_moment1_d, local_height, local_width);
      m_cudnn->copy_on_gpus(m_moment1_d, other.m_moment1_d, local_height, local_width);
      m_cudnn->allocate_on_gpus(m_moment2_d, local_height, local_width);
      m_cudnn->copy_on_gpus(m_moment2_d, other.m_moment2_d, local_height, local_width);
#endif
    }
  } else {
    m_moment1 = nullptr;
    m_moment2 = nullptr;
  }
  return *this;
}

adam::~adam() {
  if(m_moment1) {
    delete m_moment1;
    delete m_moment2;
#ifdef __LIB_CUDNN
    m_cudnn->deallocate_on_gpus(m_moment1_d);
    m_cudnn->deallocate_on_gpus(m_moment2_d);
#endif
  }
}

void adam::setup(AbsDistMat *parameters) {
  optimizer::setup(parameters);

  // Initialize Adam cache
  switch(m_matrix_format) {
  case matrix_format::MC_MR:
    m_moment1 = new DistMat(m_comm->get_model_grid());
    m_moment2 = new DistMat(m_comm->get_model_grid());
    break;
  case matrix_format::STAR_STAR:
    m_moment1 = new StarMat(m_comm->get_model_grid());
    m_moment2 = new StarMat(m_comm->get_model_grid());
    break;
  case matrix_format::MC_STAR:
    m_moment1 = new RowSumMat(m_comm->get_model_grid());
    m_moment2 = new RowSumMat(m_comm->get_model_grid());
    break;
  case matrix_format::STAR_VC:
    m_moment1 = new StarVCMat(m_comm->get_model_grid());
    m_moment2 = new StarVCMat(m_comm->get_model_grid());
    break;
  default:
    throw lbann_exception("lbann_optimizer_adam: invalid data layout");
  }
  El::Zeros(*m_moment1, m_height, m_width);
  El::Zeros(*m_moment2, m_height, m_width);
}

void adam::setup_gpu(AbsDistMat *parameters,
                     const std::vector<DataType *> &parameters_d) {
#ifdef __LIB_CUDA
  optimizer::setup_gpu(parameters, parameters_d);
  int local_height = m_parameters->LocalHeight();
  int local_width = m_parameters->LocalWidth();  
  m_cudnn->allocate_on_gpus(m_moment1_d, local_height, local_width);
  m_cudnn->clear_on_gpus(m_moment1_d, local_height, local_width);  
  m_cudnn->allocate_on_gpus(m_moment2_d, local_height, local_width);
  m_cudnn->clear_on_gpus(m_moment2_d, local_height, local_width);
#endif  
}

void adam::update(const AbsDistMat *gradient) {
  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  // Precompute the bias correction and learning rate.
  const DataType correction = m_learning_rate *
                              (std::sqrt(DataType(1) - m_current_beta2)
                               / (DataType(1) - m_current_beta1));

  // Get local matrix data
  const int local_height = m_parameters->LocalHeight();
  const int local_width = m_parameters->LocalWidth();
  DataType *parameters_buffer = m_parameters->Buffer();
  const int parameters_ldim = m_parameters->LDim();
  const DataType *gradient_buffer = gradient->LockedBuffer();
  const int gradient_ldim = gradient->LDim();
  DataType *moment1_buffer = m_moment1->Buffer();
  const int moment1_ldim = m_moment1->LDim();
  DataType *moment2_buffer = m_moment2->Buffer();
  const int moment2_ldim = m_moment2->LDim();

  // Check if matrix data is contiguous
  if(parameters_ldim != local_height
      || gradient_ldim != local_height
      || moment1_ldim != local_height
      || moment2_ldim != local_height) {
    // Update with non-contiguous data
    #pragma omp parallel for collapse(2)
    for(int j=0; j<local_width; ++j) {
      for(int i=0; i<local_height; ++i) {
        DataType& x = parameters_buffer[i+j*parameters_ldim];
        // See below; avoid denormalization.
        const DataType g = gradient_buffer[i+j*gradient_ldim] + m_eps;
        DataType& m1 = moment1_buffer[i+j*moment1_ldim];
        DataType& m2 = moment2_buffer[i+j*moment2_ldim];
        m1 = m_beta1 * m1 + (DataType(1) - m_beta1) * g;
        m2 = m_beta2 * m2 + (DataType(1) - m_beta2) * g * g;
        x -= correction * m1 / (std::sqrt(m2) + m_eps);
      }
    }
  } else {
    // Update with contiguous data
    #pragma omp parallel for
    for(int i=0; i<local_height*local_width; ++i) {
      DataType& x = parameters_buffer[i];
      // We add eps here because sometimes the gradient is small enough that
      // g*g can become denormalized, which can significantly impact the
      // performance.
      const DataType g = gradient_buffer[i] + m_eps;
      DataType& m1 = moment1_buffer[i];
      DataType& m2 = moment2_buffer[i];
      m1 = m_beta1 * m1 + (DataType(1) - m_beta1) * g;
      m2 = m_beta2 * m2 + (DataType(1) - m_beta2) * g * g;
      x -= correction * m1 / (std::sqrt(m2) + m_eps);
    }
  }
}


adam_factory::adam_factory(lbann_comm *comm, DataType learning_rate,
                           DataType beta1, DataType beta2, DataType eps,
                           cudnn::cudnn_manager *cudnn)
  : optimizer_factory(comm, "adam"),
    m_learning_rate(learning_rate),
    m_beta1(beta1),
    m_beta2(beta2),
    m_eps(eps),
    m_cudnn(cudnn) {}

adam_factory::~adam_factory() {}

optimizer *adam_factory::create_optimizer() {
  return new adam(m_comm, m_learning_rate, m_beta1, m_beta2, m_eps, m_cudnn);
}

}  // namespace lbann
