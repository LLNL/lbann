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
// rmsprop .hpp .cpp - SGD with RMSprop
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/rmsprop.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

rmsprop::rmsprop(lbann_comm *comm,
                 DataType learning_rate,
                 DataType decay_rate,
                 DataType eps)
  : optimizer(comm, learning_rate),
    m_decay_rate(decay_rate),
    m_eps(eps),
    m_cache(nullptr) {}

rmsprop::rmsprop(const rmsprop& other) :
  optimizer(other),
  m_decay_rate(other.m_decay_rate),
  m_eps(other.m_eps),
  m_cache(other.m_cache) {
  if (m_cache != nullptr) { m_cache = m_cache->Copy(); }
}

rmsprop& rmsprop::operator=(const rmsprop& other) {
  optimizer::operator=(other);
  m_decay_rate = other.m_decay_rate;
  m_eps = other.m_eps;

  // Copy cache matrix
  if (m_cache != nullptr && other.m_cache != nullptr
      && m_cache->DistData() == other.m_cache->DistData()) {
    El::Copy(*other.m_cache, *m_cache);
  }
  else {
    if (m_cache != nullptr) { delete m_cache; }
    m_cache = other.m_cache;
    if (m_cache != nullptr) { m_cache = m_cache->Copy(); }
  }

  return *this;
}

rmsprop::~rmsprop() {
  if (m_cache != nullptr) { delete m_cache; }
}

std::string rmsprop::get_description() const {
  std::stringstream ss;
  ss << optimizer::get_description() << ", "
     << "decay_rate=" << m_decay_rate << ", "
     << "eps=" << m_eps;
  return ss.str();
}

void rmsprop::setup(weights& w) {
  optimizer::setup(w);
  m_cache = m_gradient->Construct(m_gradient->Grid(),
                                  m_gradient->Root());
  El::Zeros(*m_cache, m_gradient->Height(), m_gradient->Width());
}

void rmsprop::step_compute(AbsDistMat& values, const AbsDistMat& gradient) {

  // Get local matrix data
  const int local_height = values.LocalHeight();
  const int local_width = values.LocalWidth();
  DataType* __restrict__ values_buffer = values.Buffer();
  const int values_ldim = values.LDim();
  const DataType* __restrict__ gradient_buffer = gradient.LockedBuffer();
  const int gradient_ldim = gradient.LDim();
  DataType* __restrict__ cache_buffer = m_cache->Buffer();
  const int cache_ldim = m_cache->LDim();

  // Check if matrix data is contiguous
  if (values_ldim != local_height
      || gradient_ldim != local_height
      || cache_ldim != local_height) {
    // Update with non-contiguous data
    #pragma omp parallel for collapse(2)
    for (int j=0; j<local_width; ++j) {
      for (int i=0; i<local_height; ++i) {
        DataType& x = values_buffer[i+j*values_ldim];
        const DataType g = gradient_buffer[i+j*gradient_ldim];
        DataType& c = cache_buffer[i+j*cache_ldim];
        c = m_decay_rate * c + (DataType(1) - m_decay_rate) * g * g;
        x -= m_learning_rate * g / (Sqrt(c) + m_eps);
      }
    }
  } else {
    // Update with contiguous data
    #pragma omp parallel for
    for (int i=0; i<local_height*local_width; ++i) {
      DataType& x = values_buffer[i];
      const DataType g = gradient_buffer[i];
      DataType& c = cache_buffer[i];
      c = m_decay_rate * c + (DataType(1) - m_decay_rate) * g * g;
      x -= m_learning_rate * g / (Sqrt(c) + m_eps);
    }
  }
}

}  // namespace lbann
