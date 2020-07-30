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

#include "lbann/optimizers/rmsprop.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"

namespace lbann {

template <typename TensorDataType>
rmsprop<TensorDataType>::rmsprop(TensorDataType learning_rate,
                 TensorDataType decay_rate,
                 TensorDataType eps)
  : BaseType(learning_rate),
    m_decay_rate(decay_rate),
    m_eps(eps) {}

template <typename TensorDataType>
rmsprop<TensorDataType>::rmsprop(const rmsprop& other) :
  BaseType(other),
  m_decay_rate(other.m_decay_rate),
  m_eps(other.m_eps),
  m_cache(other.m_cache ? other.m_cache->Copy() : nullptr) {}

template <typename TensorDataType>
rmsprop<TensorDataType>& rmsprop<TensorDataType>::operator=(const rmsprop& other) {
  OptimizerType::operator=(other);
  m_decay_rate = other.m_decay_rate;
  m_eps = other.m_eps;
  m_cache.reset(other.m_cache ? other.m_cache->Copy() : nullptr);
  return *this;
}

template <typename TensorDataType>
description rmsprop<TensorDataType>::get_description() const {
  auto desc = OptimizerType::get_description();
  desc.add("Decay rate", m_decay_rate);
  desc.add("eps", m_eps);
  return desc;
}

template <typename TensorDataType>
void rmsprop<TensorDataType>::setup(WeightsType* w) {
  OptimizerType::setup(w);
  const auto& gradient = this->get_gradient();
  m_cache.reset(AbsDistMatrixType::Instantiate(gradient.DistData()));
  El::Zeros(*m_cache, gradient.Height(), gradient.Width());
}

template <typename TensorDataType>
void rmsprop<TensorDataType>::step_compute(AbsDistMatrixType& values,
                                           const AbsDistMatrixType& gradient) {
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

template <typename TensorDataType>
void rmsprop<TensorDataType>::step_compute_cpu(AbsDistMatrixType& values,
                                               const AbsDistMatrixType& gradient) {

  // Get local matrix data
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  auto* __restrict__ values_buffer = values.Buffer();
  const size_t values_ldim = values.LDim();
  const auto* __restrict__ gradient_buffer = gradient.LockedBuffer();
  const size_t gradient_ldim = gradient.LDim();
  auto* __restrict__ cache_buffer = m_cache->Buffer();
  const size_t cache_ldim = m_cache->LDim();

  // Apply RMSprop step
  const auto& learning_rate = this->get_learning_rate();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (size_t col = 0; col < local_width; ++col) {
    for (size_t row = 0; row < local_height; ++row) {
      auto& x = values_buffer[row+col*values_ldim];
      const auto& g = gradient_buffer[row+col*gradient_ldim];
      auto& c = cache_buffer[row+col*cache_ldim];
      c = m_decay_rate * c + (TensorDataType(1.) - m_decay_rate) * g * g;
      x -= learning_rate * g / (El::Sqrt(c) + m_eps);
    }
  }

}

// =============================================
// Checkpointing
// =============================================

template <typename TensorDataType>
bool rmsprop<TensorDataType>::save_to_checkpoint_shared(persist& p, std::string name_prefix) {
  if (this->get_comm().am_trainer_master()) {
    write_cereal_archive(*this, p, "rmsprop.xml");
  }

  char l_name[512];
  sprintf(l_name, "%s_optimizer_cache_%lldx%lld", name_prefix.c_str(), m_cache->Height(), m_cache->Width());
  p.write_distmat(persist_type::train, l_name, m_cache.get());

  return true;
}

template <typename TensorDataType>
bool rmsprop<TensorDataType>::load_from_checkpoint_shared(persist& p, std::string name_prefix) {
  load_from_shared_cereal_archive(*this, p, this->get_comm(), "rmsprop.xml");

  char l_name[512];
  sprintf(l_name, "%s_optimizer_cache_%lldx%lld.bin", name_prefix.c_str(), m_cache->Height(), m_cache->Width());
  p.read_distmat(persist_type::train, l_name, m_cache.get());

  return true;
}

template <typename TensorDataType>
bool rmsprop<TensorDataType>::save_to_checkpoint_distributed(persist& p, std::string name_prefix) {
  write_cereal_archive(*this, p, "rmsprop.xml");

  char l_name[512];
  sprintf(l_name, "%s_optimizer_cache_%lldx%lld", name_prefix.c_str(), m_cache->Height(), m_cache->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_cache);

  return true;
}

template <typename TensorDataType>
bool rmsprop<TensorDataType>::load_from_checkpoint_distributed(persist& p, std::string name_prefix) {
  read_cereal_archive(*this, p, "rmsprop.xml");

  char l_name[512];
  sprintf(l_name, "%s_optimizer_cache_%lldx%lld", name_prefix.c_str(), m_cache->Height(), m_cache->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_cache);

  return true;
}

template <typename TensorDataType>
std::unique_ptr<optimizer>
build_rmsprop_optimizer_from_pbuf(
  google::protobuf::Message const& msg) {
  const auto& params =
    dynamic_cast<lbann_data::Optimizer::RMSprop const&>(msg);
  return make_unique<rmsprop<TensorDataType>>(
    TensorDataType(params.learn_rate()),
    TensorDataType(params.decay_rate()),
    TensorDataType(params.eps()));
}

#define PROTO(T)                                    \
  template class rmsprop<T>;                        \
  template std::unique_ptr<optimizer>               \
  build_rmsprop_optimizer_from_pbuf<T>(             \
    google::protobuf::Message const&)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
