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
// adam .hpp .cpp .cu - SGD with Adam
// Reference:
// Kingma, D. and Ba, J. 2014. Adam: A Method for Stochastic Optimization.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/adam.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

adam::adam(lbann_comm *comm,
           DataType learning_rate,
           DataType beta1,
           DataType beta2,
           DataType eps,
           cudnn::cudnn_manager *cudnn)
  : optimizer(comm, learning_rate),
    m_beta1(beta1),
    m_beta2(beta2),
    m_eps(eps),
    m_current_beta1(1),
    m_current_beta2(1),
    m_moment1(nullptr),
    m_moment2(nullptr) {}

adam::adam(const adam& other)
  : optimizer(other),
    m_beta1(other.m_beta1),
    m_beta2(other.m_beta2),
    m_eps(other.m_eps),
    m_current_beta1(other.m_current_beta1),
    m_current_beta2(other.m_current_beta2),
    m_moment1(other.m_moment1),
    m_moment2(other.m_moment2) {
  if (m_moment1 != nullptr) { m_moment1 = m_moment1->Copy(); }
  if (m_moment2 != nullptr) { m_moment2 = m_moment2->Copy(); }
  #ifdef __LIB_CUDNN
  if (m_cudnn != nullptr && other.m_weights != nullptr) {
    const int height = other.m_weights->get_height();
    const int width = other.m_weights->get_width();
    m_moment1_d = m_cudnn->copy(other.m_moment1_d, height, width);
    m_moment2_d = m_cudnn->copy(other.m_moment2_d, height, width);
  }
  #endif // __LIB_CUDNN
}

adam& adam::operator=(const adam& other) {
  optimizer::operator=(other);
  m_beta1 = other.m_beta1;
  m_beta2 = other.m_beta2;
  m_eps = other.m_eps;
  m_current_beta1 = other.m_current_beta1;
  m_current_beta2 = other.m_current_beta2;

  // Copy moment matrices
  if (m_moment1 != nullptr && other.m_moment1 != nullptr
      && m_moment1->DistData() == other.m_moment1->DistData()) {
    El::Copy(*other.m_moment1, *m_moment1);
  }
  else {
    if (m_moment1 != nullptr) { delete m_moment1; }
    m_moment1 = other.m_moment1;
    if (m_moment1 != nullptr) { m_moment1 = m_moment1->Copy(); }
  }
  if (m_moment2 != nullptr && other.m_moment2 != nullptr
      && m_moment2->DistData() == other.m_moment2->DistData()) {
    El::Copy(*other.m_moment2, *m_moment2);
  }
  else {
    if (m_moment2 != nullptr) { delete m_moment2; }
    m_moment2 = other.m_moment2;
    if (m_moment2 != nullptr) { m_moment2 = m_moment2->Copy(); }
  }

  // Copy GPU data
  #ifdef __LIB_CUDNN
  if (m_cudnn != nullptr && other.m_weights != nullptr) {
    const int height = other.m_weights->get_height();
    const int width = other.m_weights->get_width();
    m_cudnn->deallocate_on_gpus(m_moment1_d);
    m_cudnn->deallocate_on_gpus(m_moment2_d);
    m_moment1_d = m_cudnn->copy(other.m_moment1_d, height, width);
    m_moment2_d = m_cudnn->copy(other.m_moment2_d, height, width);
  }
  #endif // __LIB_CUDNN

  return *this;
}

adam::~adam() {
  if(m_moment1 != nullptr) { delete m_moment1; }
  if(m_moment2 != nullptr) { delete m_moment2; }
  #ifdef __LIB_CUDNN
  if (m_cudnn != nullptr) {
    m_cudnn->deallocate_on_gpus(m_moment1_d);
    m_cudnn->deallocate_on_gpus(m_moment2_d);
  }
  #endif // __LIB_CUDNN
}

std::string adam::get_description() const {
  std::stringstream ss;
  ss << optimizer::get_description() << ", "
     << "beta1=" << m_beta1 << ", "
     << "beta2=" << m_beta2 << ", "
     << "eps=" << m_eps;
  return ss.str();
}

void adam::setup(weights& w) {
  optimizer::setup(w);

  const int height = m_gradient->Height();
  const int width = m_gradient->Width();

  m_moment1 = m_gradient->Construct(m_gradient->Grid(),
                                    m_gradient->Root());
  m_moment2 = m_gradient->Construct(m_gradient->Grid(),
                                    m_gradient->Root());
  El::Zeros(*m_moment1, height, width);
  El::Zeros(*m_moment2, height, width);
  
  if (m_cudnn != nullptr) {
#ifdef __LIB_CUDNN
    m_cudnn->allocate_on_gpus(m_moment1_d, height, width);
    m_cudnn->allocate_on_gpus(m_moment2_d, height, width);
#endif // __LIB_CUDNN
  }

}

void adam::step_compute(AbsDistMat& values, const AbsDistMat& gradient) {

  // Precompute the bias correction and learning rate.
  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  const DataType correction = m_learning_rate *
                              (std::sqrt(DataType(1) - m_current_beta2)
                               / (DataType(1) - m_current_beta1));

  // Get local matrix data
  const int local_height = values.LocalHeight();
  const int local_width = values.LocalWidth();
  DataType* __restrict__ values_buffer = values.Buffer();
  const int values_ldim = values.LDim();
  const DataType* __restrict__ gradient_buffer = gradient.LockedBuffer();
  const int gradient_ldim = gradient.LDim();
  DataType* __restrict__ moment1_buffer = m_moment1->Buffer();
  const int moment1_ldim = m_moment1->LDim();
  DataType* __restrict__ moment2_buffer = m_moment2->Buffer();
  const int moment2_ldim = m_moment2->LDim();

  // Check if matrix data is contiguous
  if (values_ldim != local_height
      || gradient_ldim != local_height
      || moment1_ldim != local_height
      || moment2_ldim != local_height) {
    // Update with non-contiguous data
    #pragma omp parallel for collapse(2)
    for (int j=0; j<local_width; ++j) {
      for (int i=0; i<local_height; ++i) {
        DataType& x = values_buffer[i+j*values_ldim];
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
    for (int i=0; i<local_height*local_width; ++i) {
      DataType& x = values_buffer[i];
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

}  // namespace lbann
