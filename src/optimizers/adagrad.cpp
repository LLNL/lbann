////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
  : optimizer(comm, learning_rate), m_eps(eps) {}

adagrad::adagrad(const adagrad& other)
  : optimizer(other),
    m_eps(other.m_eps),
    m_cache(other.m_cache ? other.m_cache->Copy() : nullptr) {}

adagrad& adagrad::operator=(const adagrad& other) {
  optimizer::operator=(other);
  m_eps = other.m_eps;
  m_cache.reset(other.m_cache ? other.m_cache->Copy() : nullptr);
  return *this;
}

description adagrad::get_description() const {
  auto&& desc = optimizer::get_description();
  desc.add("eps", m_eps);
  return desc;
}

void adagrad::setup(weights* w) {
  optimizer::setup(w);
  const auto& gradient = this->get_gradient();
  m_cache.reset(AbsDistMat::Instantiate(gradient.DistData()));
  El::Zeros(*m_cache, gradient.Height(), gradient.Width());
}

void adagrad::step_compute(AbsDistMat& values, const AbsDistMat& gradient) {
  switch (values.GetLocalDevice()) {
  case El::Device::CPU: step_compute_cpu(values, gradient); break;
#ifdef LBANN_HAS_CUDA
  case El::Device::GPU: step_compute_gpu(values, gradient); break;
#endif // LBANN_HAS_CUDA
  default:
    std::ostringstream err;
    err << "unsupported device type "
        << "(" << static_cast<int>(values.GetLocalDevice()) << ")";
    LBANN_ERROR(err.str());
  }
}

void adagrad::step_compute_cpu(AbsDistMat& values, const AbsDistMat& gradient) {

  // Get local matrix data
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  auto* __restrict__ values_buffer = values.Buffer();
  const size_t values_ldim = values.LDim();
  const auto* __restrict__ gradient_buffer = gradient.LockedBuffer();
  const size_t gradient_ldim = gradient.LDim();
  auto* __restrict__ cache_buffer = m_cache->Buffer();
  const size_t cache_ldim = m_cache->LDim();

  // Apply AdaGrad step
  const auto& learning_rate = get_learning_rate();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (size_t col = 0; col < local_width; ++col) {
    for (size_t row = 0; row < local_height; ++row) {
      auto& x = values_buffer[row+col*values_ldim];
      const auto& g = gradient_buffer[row+col*gradient_ldim];
      auto& c = cache_buffer[row+col*cache_ldim];
      c += g * g;
      x -= learning_rate * g / (std::sqrt(c) + m_eps);
    }
  }

}

// =============================================
// Checkpointing
// =============================================

bool adagrad::save_to_checkpoint_shared(persist& p, std::string name_prefix) {
  optimizer::save_to_checkpoint_shared(p, name_prefix);

  char l_name[512];
  sprintf(l_name, "%s_optimizer_cache_%lldx%lld", name_prefix.c_str(), m_cache->Height(), m_cache->Width());
  p.write_distmat(persist_type::train, l_name, m_cache.get());

  return true;
}

bool adagrad::load_from_checkpoint_shared(persist& p, std::string name_prefix) {
  optimizer::load_from_checkpoint_shared(p, name_prefix);
  char l_name[512];

  sprintf(l_name, "%s_optimizer_cache_%lldx%lld.bin", name_prefix.c_str(), m_cache->Height(), m_cache->Width());
  p.read_distmat(persist_type::train, l_name, m_cache.get());

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

} // namespace lbann
