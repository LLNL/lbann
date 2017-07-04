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
// lbann_optimizer_hypergradient_adam .hpp .cpp - Hypergradient SGD with Adam
// Reference:
// Baydin et al. "Online Learning Rate Adaptation with Hypergradient Descent", 2017.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/optimizer_hypergradient_adam.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

hypergradient_adam::hypergradient_adam(
  lbann_comm *comm,
  DataType init_learning_rate,
  DataType hyper_learning_rate,
  DataType beta1,
  DataType beta2,
  DataType eps) :
  optimizer(comm, init_learning_rate),
  m_hyper_learning_rate(hyper_learning_rate),
  m_beta1(beta1), m_beta2(beta2), m_eps(eps), m_current_beta1(1),
  m_current_beta2(1), m_moment1(nullptr), m_moment2(nullptr),
  m_old_gradient(nullptr) {}

hypergradient_adam::hypergradient_adam(const hypergradient_adam& other) :
  optimizer(other), m_hyper_learning_rate(other.m_hyper_learning_rate),
  m_beta1(other.m_beta1), m_beta2(other.m_beta2), m_eps(other.m_eps),
  m_current_beta1(other.m_current_beta1),
  m_current_beta2(other.m_current_beta2), m_moment1(nullptr),
  m_moment2(nullptr), m_old_gradient(nullptr) {
  if (other.m_moment1) {
    m_moment1 = other.m_moment1->Copy();
    m_moment2 = other.m_moment2->Copy();
    m_old_gradient = other.m_old_gradient->Copy();
  }
}

hypergradient_adam& hypergradient_adam::operator=(
  const hypergradient_adam& other) {
  optimizer::operator=(other);
  m_beta1 = other.m_beta1;
  m_beta2 = other.m_beta2;
  m_eps = other.m_eps;
  m_current_beta1 = other.m_current_beta1;
  m_current_beta2 = other.m_current_beta2;
  if (other.m_moment1) {
    m_moment1 = other.m_moment1->Copy();
    m_moment2 = other.m_moment2->Copy();
    m_old_gradient = other.m_old_gradient->Copy();
  } else {
    m_moment1 = nullptr;
    m_moment2 = nullptr;
    m_old_gradient = nullptr;
  }
  return *this;
}

hypergradient_adam::~hypergradient_adam() {
  if (m_moment1) {
    delete m_moment1;
    delete m_moment2;
    delete m_old_gradient;
  }
}

void hypergradient_adam::setup(AbsDistMat *parameters) {
  optimizer::setup(parameters);

  // Initialize running averages.
  switch (m_matrix_format) {
  case matrix_format::MC_MR:
    m_moment1 = new DistMat(m_comm->get_model_grid());
    m_moment2 = new DistMat(m_comm->get_model_grid());
    m_old_gradient = new DistMat(m_comm->get_model_grid());
    break;
  case matrix_format::STAR_STAR:
    m_moment1 = new StarMat(m_comm->get_model_grid());
    m_moment2 = new StarMat(m_comm->get_model_grid());
    m_old_gradient = new StarMat(m_comm->get_model_grid());
    break;
  case matrix_format::MC_STAR:
    m_moment1 = new RowSumMat(m_comm->get_model_grid());
    m_moment2 = new RowSumMat(m_comm->get_model_grid());
    m_old_gradient = new RowSumMat(m_comm->get_model_grid());
    break;
  case matrix_format::STAR_VC:
    m_moment1 = new StarVCMat(m_comm->get_model_grid());
    m_moment2 = new StarVCMat(m_comm->get_model_grid());
    m_old_gradient = new StarVCMat(m_comm->get_model_grid());
    break;
  default:
    throw lbann_exception(
      "lbann_optimizer_hypergradient_adam: invalid data layout");
  }
  El::Zeros(*m_moment1, m_height, m_width);
  El::Zeros(*m_moment2, m_height, m_width);
  El::Zeros(*m_old_gradient, m_height, m_width);
}

void hypergradient_adam::update(const AbsDistMat *gradient) {
  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  // Precompute the bias correction.
  const DataType correction = std::sqrt(DataType(1) - m_current_beta2) /
                              (DataType(1) - m_current_beta1);

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
  DataType *old_gradient_buffer = m_old_gradient->Buffer();
  const int old_gradient_ldim = m_old_gradient->LDim();

  // Compute the learning rate update.
  DataType lr_update = El::Dot(*gradient, *m_old_gradient);
  m_learning_rate += m_hyper_learning_rate * lr_update;

  // Check if matrix data is contiguous.
  if (parameters_ldim != local_height
      || gradient_ldim != local_height
      || moment1_ldim != local_height
      || moment2_ldim != local_height
      || old_gradient_ldim != local_height) {
    // Non-contiguous data.
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < local_width; ++j) {
      for (int i = 0; i < local_height; ++i) {
        DataType& x = parameters_buffer[i+j*parameters_ldim];
        const DataType g = gradient_buffer[i+j*gradient_ldim] + m_eps;
        DataType& m1 = moment1_buffer[i+j*moment1_ldim];
        DataType& m2 = moment2_buffer[i+j*moment2_ldim];
        DataType& old_c = old_gradient_buffer[i+j*old_gradient_ldim];
        m1 = m_beta1 * m1 + (DataType(1) - m_beta1) * g;
        m2 = m_beta2 * m2 + (DataType(1) - m_beta2) * g * g;
        old_c = correction * m1 / (std::sqrt(m2) + m_eps);
        x -= m_learning_rate * old_c;
      }
    }
  } else {
    // Contiguous data.
    #pragma omp parallel for
    for (int i = 0; i < local_height * local_width; ++i) {
      DataType& x = parameters_buffer[i];
      // Add eps here to avoid denormalized floats.
      const DataType g = gradient_buffer[i] + m_eps;
      DataType& m1 = moment1_buffer[i];
      DataType& m2 = moment2_buffer[i];
      DataType& old_c = old_gradient_buffer[i];
      // Update the first/second moment estimates.
      m1 = m_beta1 * m1 + (DataType(1) - m_beta1) * g;
      m2 = m_beta2 * m2 + (DataType(1) - m_beta2) * g * g;
      // Compute the unbiased gradient estimate.
      old_c = correction * m1 / (std::sqrt(m2) + m_eps);
      // Parameter update.
      x -= m_learning_rate * old_c;
    }
  }
}

hypergradient_adam_factory::hypergradient_adam_factory(
  lbann_comm *comm,
  DataType init_learning_rate, DataType hyper_learning_rate,
  DataType beta1, DataType beta2, DataType eps) :
  optimizer_factory(comm, "hypergradient_adam"),
  m_learning_rate(init_learning_rate),
  m_hyper_learning_rate(hyper_learning_rate),
  m_beta1(beta1),
  m_beta2(beta2),
  m_eps(eps) {}

hypergradient_adam_factory::~hypergradient_adam_factory() {}

optimizer *hypergradient_adam_factory::create_optimizer() {
  return new hypergradient_adam(m_comm, m_learning_rate, m_hyper_learning_rate,
                                m_beta1, m_beta2, m_eps);
}

}  // namespace lbann
