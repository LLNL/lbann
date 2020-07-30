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

#include "lbann/optimizers/hypergradient_adam.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"

#include <optimizers.pb.h>

namespace lbann {

template <typename TensorDataType>
hypergradient_adam<TensorDataType>::hypergradient_adam(TensorDataType init_learning_rate,
                                                       TensorDataType hyper_learning_rate,
                                                       TensorDataType beta1,
                                                       TensorDataType beta2,
                                                       TensorDataType eps)
  : BaseType(init_learning_rate),
    m_hyper_learning_rate(hyper_learning_rate),
    m_beta1(beta1),
    m_beta2(beta2),
    m_eps(eps),
    m_current_beta1(1.),
    m_current_beta2(1.) {}

template <typename TensorDataType>
hypergradient_adam<TensorDataType>::hypergradient_adam(const hypergradient_adam& other)
  : BaseType(other),
    m_hyper_learning_rate(other.m_hyper_learning_rate),
    m_beta1(other.m_beta1),
    m_beta2(other.m_beta2),
    m_eps(other.m_eps),
    m_current_beta1(other.m_current_beta1),
    m_current_beta2(other.m_current_beta2),
    m_moment1(other.m_moment1 ? other.m_moment1->Copy() : nullptr),
    m_moment2(other.m_moment2 ? other.m_moment2->Copy() : nullptr),
    m_old_gradient(other.m_old_gradient ?
                   other.m_old_gradient->Copy() : nullptr) {}

template <typename TensorDataType>
hypergradient_adam<TensorDataType>& hypergradient_adam<TensorDataType>::operator=(const hypergradient_adam<TensorDataType>& other) {
  OptimizerType::operator=(other);
  m_hyper_learning_rate = other.m_hyper_learning_rate;
  m_beta1 = other.m_beta1;
  m_beta2 = other.m_beta2;
  m_eps = other.m_eps;
  m_current_beta1 = other.m_current_beta1;
  m_current_beta2 = other.m_current_beta2;
  m_moment1.reset(other.m_moment1 ? other.m_moment1->Copy() : nullptr);
  m_moment2.reset(other.m_moment2 ? other.m_moment2->Copy() : nullptr);
  m_old_gradient.reset(other.m_old_gradient ?
                       other.m_old_gradient->Copy() : nullptr);
  return *this;
}

template <typename TensorDataType>
description hypergradient_adam<TensorDataType>::get_description() const {
  auto desc = OptimizerType::get_description();
  desc.add("Hypergradient learning rate", m_hyper_learning_rate);
  desc.add("beta1", m_beta1);
  desc.add("beta2", m_beta2);
  desc.add("eps", m_eps);
  return desc;
}

template <typename TensorDataType>
void hypergradient_adam<TensorDataType>::setup(WeightsType* w) {
  OptimizerType::setup(w);
  const auto& gradient = this->get_gradient();
  m_moment1.reset(AbsDistMatrixType::Instantiate(gradient.DistData()));
  m_moment2.reset(AbsDistMatrixType::Instantiate(gradient.DistData()));
  m_old_gradient.reset(AbsDistMatrixType::Instantiate(gradient.DistData()));
  El::Zeros(*m_moment1, gradient.Height(), gradient.Width());
  El::Zeros(*m_moment2, gradient.Height(), gradient.Width());
  El::Zeros(*m_old_gradient, gradient.Height(), gradient.Width());
}

template <typename TensorDataType>
void hypergradient_adam<TensorDataType>::step_compute(AbsDistMatrixType& values,
                                                      const AbsDistMatrixType& gradient) {
  if (values.GetLocalDevice() != El::Device::CPU) {
    LBANN_ERROR("hypergradient Adam is only supported on CPU");
  }

  // Precompute the bias correction.
  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  const TensorDataType correction = El::Sqrt(TensorDataType(1.) - m_current_beta2) /
                              (TensorDataType(1.) - m_current_beta1);

  // Get local matrix data
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  auto* __restrict__ values_buffer = values.Buffer();
  const size_t values_ldim = values.LDim();
  const TensorDataType* __restrict__ gradient_buffer = gradient.LockedBuffer();
  const size_t gradient_ldim = gradient.LDim();
  auto* __restrict__ moment1_buffer = m_moment1->Buffer();
  const size_t moment1_ldim = m_moment1->LDim();
  auto* __restrict__ moment2_buffer = m_moment2->Buffer();
  const size_t moment2_ldim = m_moment2->LDim();
  auto* __restrict__ old_gradient_buffer = m_old_gradient->Buffer();
  const size_t old_gradient_ldim = m_old_gradient->LDim();

  // Compute the learning rate update.
  TensorDataType lr_update = El::Dot(gradient, *m_old_gradient);
  auto learning_rate = this->get_learning_rate();
  learning_rate += m_hyper_learning_rate * lr_update;
  this->set_learning_rate(learning_rate);

  // Hypergradient Adam step
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (size_t col = 0; col < local_width; ++col) {
    for (size_t row = 0; row < local_height; ++row) {
      auto& x = values_buffer[row+col*values_ldim];
      const auto g = gradient_buffer[row+col*gradient_ldim] + m_eps;
      auto& m1 = moment1_buffer[row+col*moment1_ldim];
      auto& m2 = moment2_buffer[row+col*moment2_ldim];
      auto& old_c = old_gradient_buffer[row+col*old_gradient_ldim];
      m1 = m_beta1 * m1 + (TensorDataType(1.) - m_beta1) * g;
      m2 = m_beta2 * m2 + (TensorDataType(1.) - m_beta2) * g * g;
      old_c = correction * m1 / (El::Sqrt(m2) + m_eps);
      x -= learning_rate * old_c;
    }
  }

}

template <typename TensorDataType>
bool hypergradient_adam<TensorDataType>::save_to_checkpoint_shared(persist& p, std::string name_prefix) {
  if (this->get_comm().am_trainer_master()) {
    write_cereal_archive(*this, p, "hypergradient_adam.xml");
  }

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.write_distmat(persist_type::train, l_name, m_moment1.get());

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.write_distmat(persist_type::train, l_name, m_moment2.get());

  sprintf(l_name, "%s_optimizer_adam_old_gradient_%lldx%lld", name_prefix.c_str(), m_old_gradient->Height(), m_old_gradient->Width());
  p.write_distmat(persist_type::train, l_name, m_old_gradient.get());

  return true;
}

template <typename TensorDataType>
bool hypergradient_adam<TensorDataType>::load_from_checkpoint_shared(persist& p, std::string name_prefix) {
  load_from_shared_cereal_archive(*this, p, this->get_comm(), "hypergradient_adam.xml");

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld.bin", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.read_distmat(persist_type::train, l_name, m_moment1.get());

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld.bin", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.read_distmat(persist_type::train, l_name, m_moment2.get());

  sprintf(l_name, "%s_optimizer_adam_old_gradient_%lldx%lld.bin", name_prefix.c_str(), m_old_gradient->Height(), m_old_gradient->Width());
  p.read_distmat(persist_type::train, l_name, m_old_gradient.get());
  return true;
}

template <typename TensorDataType>
bool hypergradient_adam<TensorDataType>::save_to_checkpoint_distributed(persist& p, std::string name_prefix) {
  write_cereal_archive(*this, p, "hypergradient_adam.xml");

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_moment1);

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_moment2);

  sprintf(l_name, "%s_optimizer_adam_old_gradient_%lldx%lld", name_prefix.c_str(), m_old_gradient->Height(), m_old_gradient->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_old_gradient);

  return true;
}

template <typename TensorDataType>
bool hypergradient_adam<TensorDataType>::load_from_checkpoint_distributed(persist& p, std::string name_prefix) {
  read_cereal_archive(*this, p, "hypergradient_adam.xml");

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_moment1);

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_moment2);

  sprintf(l_name, "%s_optimizer_adam_old_gradient_%lldx%lld", name_prefix.c_str(), m_old_gradient->Height(), m_old_gradient->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_old_gradient);
  return true;
}

template <typename TensorDataType>
std::unique_ptr<optimizer>
build_hypergradient_adam_optimizer_from_pbuf(
  google::protobuf::Message const& msg) {
  const auto& params =
    dynamic_cast<lbann_data::Optimizer::HypergradientAdam const&>(msg);
  return make_unique<hypergradient_adam<TensorDataType>>(
    TensorDataType(params.init_learning_rate()),
    TensorDataType(params.hyper_learning_rate()),
    TensorDataType(params.beta1()),
    TensorDataType(params.beta2()),
    TensorDataType(params.eps()));
}

#define PROTO(T)                                    \
  template class hypergradient_adam<T>;             \
  template std::unique_ptr<optimizer>               \
  build_hypergradient_adam_optimizer_from_pbuf<T>(  \
    google::protobuf::Message const&)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace lbann
