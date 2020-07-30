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

#include "lbann/optimizers/adam.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"

namespace lbann {

template <typename TensorDataType>
adam<TensorDataType>::adam(TensorDataType learning_rate,
                           TensorDataType beta1,
                           TensorDataType beta2,
                           TensorDataType eps)
  : BaseType(learning_rate),
    m_beta1(beta1), m_beta2(beta2), m_eps(eps) {}

template <typename TensorDataType>
adam<TensorDataType>::adam(const adam& other)
  : BaseType(other),
    m_beta1(other.m_beta1),
    m_beta2(other.m_beta2),
    m_eps(other.m_eps),
    m_current_beta1(other.m_current_beta1),
    m_current_beta2(other.m_current_beta2),
    m_moment1(other.m_moment1 ? other.m_moment1->Copy() : nullptr),
    m_moment2(other.m_moment2 ? other.m_moment2->Copy() : nullptr) {}

template <typename TensorDataType>
adam<TensorDataType>& adam<TensorDataType>::operator=(const adam<TensorDataType>& other) {
  OptimizerType::operator=(other);
  m_beta1 = other.m_beta1;
  m_beta2 = other.m_beta2;
  m_eps = other.m_eps;
  m_current_beta1 = other.m_current_beta1;
  m_current_beta2 = other.m_current_beta2;
  m_moment1.reset(other.m_moment1 ?
                  other.m_moment1->Copy() : nullptr);
  m_moment2.reset(other.m_moment2 ?
                  other.m_moment2->Copy() : nullptr);
  return *this;
}

template <typename TensorDataType>
description adam<TensorDataType>::get_description() const {
  auto desc = OptimizerType::get_description();
  desc.add("beta1", m_beta1);
  desc.add("beta2", m_beta2);
  desc.add("eps", m_eps);
  return desc;
}

template <typename TensorDataType>
auto adam<TensorDataType>::get_moment1() const -> const AbsDistMatrixType& {
  if (m_moment1 == nullptr) {
    LBANN_ERROR(this->get_type() + " optimizer "
                + "attempted to access moment1 before it was setup");
  }
  return *m_moment1;
}
template <typename TensorDataType>
auto adam<TensorDataType>::get_moment1() -> AbsDistMatrixType& {
  // Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers
  return const_cast<El::AbstractDistMatrix<TensorDataType>&>(static_cast<const adam<TensorDataType>&>(*this).get_moment1());
}
template <typename TensorDataType>
auto adam<TensorDataType>::get_moment2() const -> const AbsDistMatrixType& {
  if (m_moment2 == nullptr) {
    LBANN_ERROR(this->get_type() + " optimizer "
                + "attempted to access moment2 before it was setup");
  }
  return *m_moment2;
}
template <typename TensorDataType>
auto adam<TensorDataType>::get_moment2() -> AbsDistMatrixType& {
  // Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers
  return const_cast<AbsDistMatrixType&>(static_cast<const adam<TensorDataType>&>(*this).get_moment2());
}

template <typename TensorDataType>
void adam<TensorDataType>::setup(WeightsType* w) {
  OptimizerType::setup(w);
  const auto& gradient = this->get_gradient();
  m_moment1.reset(AbsDistMatrixType::Instantiate(gradient.DistData()));
  m_moment2.reset(AbsDistMatrixType::Instantiate(gradient.DistData()));
  El::Zeros(*m_moment1, gradient.Height(), gradient.Width());
  El::Zeros(*m_moment2, gradient.Height(), gradient.Width());
}

template <typename TensorDataType>
void adam<TensorDataType>::step_compute(AbsDistMatrixType& values,
                                        const AbsDistMatrixType& gradient) {
  static const auto one = TensorDataType(1.);

  // Precompute the bias correction and learning rate.
  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  const TensorDataType correction = this->get_learning_rate() *
                              (El::Sqrt(one - m_current_beta2)
                               / (one - m_current_beta1));

  switch (values.GetLocalDevice()) {
  case El::Device::CPU: step_compute_cpu(values, gradient, correction); break;
#ifdef LBANN_HAS_CUDA
  case El::Device::GPU: step_compute_gpu(values, gradient, correction); break;
#endif // LBANN_HAS_CUDA
  default:
    std::ostringstream err;
    err << "unsupported device type "
        << "(" << static_cast<int>(values.GetLocalDevice()) << ")";
    LBANN_ERROR(err.str());
  }
}

template <typename TensorDataType>
void adam<TensorDataType>::step_compute_cpu(AbsDistMatrixType& values,
                                            const AbsDistMatrixType& gradient,
                                            const TensorDataType& correction) {
  static const auto one = TensorDataType(1.);

  // Get local matrix data
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  auto* __restrict__ values_buffer = values.Buffer();
  const auto* __restrict__ gradient_buffer = gradient.LockedBuffer();
  auto* __restrict__ moment1_buffer = m_moment1->Buffer();
  auto* __restrict__ moment2_buffer = m_moment2->Buffer();

  if (values.Contiguous() && gradient.Contiguous()
      && m_moment1->Contiguous() && m_moment2->Contiguous()) {

    // Update with contiguous data
    const size_t local_size = local_height * local_width;
    LBANN_OMP_PARALLEL_FOR
    for (size_t i = 0; i < local_size; ++i) {
      auto& x = values_buffer[i];
      const auto& g = gradient_buffer[i] + m_eps; // Avoid denormalized floats
      auto& m1 = moment1_buffer[i];
      auto& m2 = moment2_buffer[i];
      m1 = m_beta1 * m1 + (one - m_beta1) * g;
      m2 = m_beta2 * m2 + (one - m_beta2) * g * g;
      x -= correction * m1 / (El::Sqrt(m2) + m_eps);
    }

  } else {

    // Update with non-contiguous data
    const size_t values_ldim = values.LDim();
    const size_t gradient_ldim = gradient.LDim();
    const size_t moment1_ldim = m_moment1->LDim();
    const size_t moment2_ldim = m_moment2->LDim();
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t col = 0; col < local_width; ++col) {
      for (size_t row = 0; row < local_height; ++row) {
        auto& x = values_buffer[row+col*values_ldim];
        const auto& g = gradient_buffer[row+col*gradient_ldim] + m_eps; // Avoid denormalized floats
        auto& m1 = moment1_buffer[row+col*moment1_ldim];
        auto& m2 = moment2_buffer[row+col*moment2_ldim];
        m1 = m_beta1 * m1 + (one - m_beta1) * g;
        m2 = m_beta2 * m2 + (one - m_beta2) * g * g;
        x -= correction * m1 / (El::Sqrt(m2) + m_eps);
      }
    }

  }

}

// =============================================
// Checkpointing
// =============================================

template <typename TensorDataType>
bool adam<TensorDataType>::save_to_checkpoint_shared(persist& p, std::string name_prefix) {
  if (this->get_comm().am_trainer_master()) {
    write_cereal_archive(*this, p, "adam.xml");
  }

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.write_distmat(persist_type::train, l_name, m_moment1.get());

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.write_distmat(persist_type::train, l_name, m_moment2.get());

  return true;
}

template <typename TensorDataType>
bool adam<TensorDataType>::load_from_checkpoint_shared(persist& p, std::string name_prefix) {
  load_from_shared_cereal_archive(*this, p, this->get_comm(), "adam.xml");

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld.bin", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.read_distmat(persist_type::train, l_name, m_moment1.get());

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld.bin", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.read_distmat(persist_type::train, l_name, m_moment2.get());

  return true;
}

template <typename TensorDataType>
bool adam<TensorDataType>::save_to_checkpoint_distributed(persist& p, std::string name_prefix) {
  write_cereal_archive(*this, p, "adam.xml");

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_moment1);

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_moment2);

  return true;
}

template <typename TensorDataType>
bool adam<TensorDataType>::load_from_checkpoint_distributed(persist& p, std::string name_prefix) {
  read_cereal_archive(*this, p, "adam.xml");

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_moment1);

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_moment2);

  return true;
}

template <typename TensorDataType>
std::unique_ptr<optimizer>
build_adam_optimizer_from_pbuf(
  google::protobuf::Message const& msg) {
  const auto& params =
    dynamic_cast<lbann_data::Optimizer::Adam const&>(msg);
  return make_unique<adam<TensorDataType>>(TensorDataType(params.learn_rate()),
                                           TensorDataType(params.beta1()),
                                           TensorDataType(params.beta2()),
                                           TensorDataType(params.eps()));
}

#define PROTO(T)                                    \
  template class adam<T>;                           \
  template std::unique_ptr<optimizer>               \
  build_adam_optimizer_from_pbuf<T>(                \
    google::protobuf::Message const&)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
