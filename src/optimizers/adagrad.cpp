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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/adagrad.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

adagrad::adagrad(lbann_comm *comm, DataType learning_rate, DataType eps)
  : optimizer(comm, learning_rate), m_eps(eps), m_cache(nullptr) {}

adagrad::adagrad(const adagrad& other)
  : optimizer(other), m_eps(other.m_eps), m_cache(other.m_cache) {
  if (m_cache != nullptr) { m_cache = m_cache->Copy(); }
}

adagrad& adagrad::operator=(const adagrad& other) {
  optimizer::operator=(other);
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

adagrad::~adagrad() {
  if (m_cache != nullptr) { delete m_cache; }
}

std::string adagrad::get_description() const {
  std::stringstream ss;
  ss << optimizer::get_description() << ", "
     << "eps=" << m_eps;
  return ss.str();
}

void adagrad::setup(weights& w) {
  optimizer::setup(w);
  m_cache = m_gradient->Construct(m_gradient->Grid(),
                                  m_gradient->Root());
  El::Zeros(*m_cache, m_gradient->Height(), m_gradient->Width());
}

void adagrad::step_compute(AbsDistMat& values, const AbsDistMat& gradient) {

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
        c += g * g;
        x -= m_learning_rate * g / (std::sqrt(c) + m_eps);
      }
    }
  } else {
    // Update with contiguous data
    #pragma omp parallel for
    for (int i=0; i<local_height*local_width; ++i) {
      DataType& x = values_buffer[i];
      const DataType g = gradient_buffer[i];
      DataType& c = cache_buffer[i];
      c += g * g;
      x -= m_learning_rate * g / (std::sqrt(c) + m_eps);
    }
  }
}

////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

bool adagrad::save_to_checkpoint_shared(persist& p, std::string name_prefix) {
  optimizer::save_to_checkpoint_shared(p, name_prefix);
  #ifdef LBANN_HAS_HDF5
  std::string group_name = name_prefix + "_optimizer";
  p.write_hdf5_distmat(group_name, "cache", m_cache, m_comm->am_model_master());
  #else
  char l_name[512];
  sprintf(l_name, "%s_optimizer_cache_%lldx%lld", name_prefix.c_str(), m_cache->Height(), m_cache->Width());
  p.write_distmat(persist_type::train, l_name, m_cache);
  #endif
  return true;
}

bool adagrad::load_from_checkpoint_shared(persist& p, std::string name_prefix) {
  optimizer::load_from_checkpoint_shared(p, name_prefix);
  #ifdef LBANN_HAS_HDF5
  std::string group_name = name_prefix + "_optimizer";
  p.read_hdf5_distmat(group_name, "cache", m_cache, m_comm->am_model_master());
  #else
  char l_name[512];
  sprintf(l_name, "%s_optimizer_cache_%lldx%lld.bin", name_prefix.c_str(), m_cache->Height(), m_cache->Width());
  p.read_distmat(persist_type::train, l_name, m_cache);
  #endif
  return true;
}

bool adagrad::save_to_checkpoint_distributed(persist& p, std::string name_prefix) {
  optimizer::save_to_checkpoint_distributed(p, name_prefix);

  char l_name[512];
  sprintf(l_name, "%s_optimizer_cache_%lldx%lld", name_prefix.c_str(), m_cache->Height(), m_cache->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_cache);

  return true;
}

bool adagrad::load_from_checkpoint_distributed(persist& p, std::string name_prefix) {
  optimizer::load_from_checkpoint_distributed(p, name_prefix);
  char l_name[512];

  sprintf(l_name, "%s_optimizer_cache_%lldx%lld", name_prefix.c_str(), m_cache->Height(), m_cache->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_cache);

  return true;
}

}  // namespace lbann
