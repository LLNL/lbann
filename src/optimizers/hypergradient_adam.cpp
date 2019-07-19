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

namespace lbann {

hypergradient_adam::hypergradient_adam(lbann_comm *comm,
                                       DataType init_learning_rate,
                                       DataType hyper_learning_rate,
                                       DataType beta1,
                                       DataType beta2,
                                       DataType eps)
  : optimizer(comm, init_learning_rate),
    m_hyper_learning_rate(hyper_learning_rate),
    m_beta1(beta1),
    m_beta2(beta2),
    m_eps(eps),
    m_current_beta1(1),
    m_current_beta2(1) {}

hypergradient_adam::hypergradient_adam(const hypergradient_adam& other)
  : optimizer(other),
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

hypergradient_adam& hypergradient_adam::operator=(const hypergradient_adam& other) {
  optimizer::operator=(other);
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

description hypergradient_adam::get_description() const {
  auto desc = optimizer::get_description();
  desc.add("Hypergradient learning rate", m_hyper_learning_rate);
  desc.add("beta1", m_beta1);
  desc.add("beta2", m_beta2);
  desc.add("eps", m_eps);
  return desc;
}

void hypergradient_adam::setup(weights* w) {
  optimizer::setup(w);
  const auto& gradient = this->get_gradient();
  m_moment1.reset(AbsDistMat::Instantiate(gradient.DistData()));
  m_moment2.reset(AbsDistMat::Instantiate(gradient.DistData()));
  m_old_gradient.reset(AbsDistMat::Instantiate(gradient.DistData()));
  El::Zeros(*m_moment1, gradient.Height(), gradient.Width());
  El::Zeros(*m_moment2, gradient.Height(), gradient.Width());
  El::Zeros(*m_old_gradient, gradient.Height(), gradient.Width());
}

void hypergradient_adam::step_compute(AbsDistMat& values,
                                      const AbsDistMat& gradient) {
  if (values.GetLocalDevice() != El::Device::CPU) {
    LBANN_ERROR("hypergradient Adam is only supported on CPU");
  }

  // Precompute the bias correction.
  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  const DataType correction = std::sqrt(DataType(1) - m_current_beta2) /
                              (DataType(1) - m_current_beta1);

  // Get local matrix data
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  auto* __restrict__ values_buffer = values.Buffer();
  const size_t values_ldim = values.LDim();
  const DataType* __restrict__ gradient_buffer = gradient.LockedBuffer();
  const size_t gradient_ldim = gradient.LDim();
  auto* __restrict__ moment1_buffer = m_moment1->Buffer();
  const size_t moment1_ldim = m_moment1->LDim();
  auto* __restrict__ moment2_buffer = m_moment2->Buffer();
  const size_t moment2_ldim = m_moment2->LDim();
  auto* __restrict__ old_gradient_buffer = m_old_gradient->Buffer();
  const size_t old_gradient_ldim = m_old_gradient->LDim();

  // Compute the learning rate update.
  DataType lr_update = El::Dot(gradient, *m_old_gradient);
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
      m1 = m_beta1 * m1 + (DataType(1) - m_beta1) * g;
      m2 = m_beta2 * m2 + (DataType(1) - m_beta2) * g * g;
      old_c = correction * m1 / (std::sqrt(m2) + m_eps);
      x -= learning_rate * old_c;
    }
  }

}

bool hypergradient_adam::save_to_checkpoint_shared(persist& p, std::string name_prefix) {
  if(p.get_cb_type() == callback_type::batch)
    optimizer::save_to_checkpoint_shared(p,name_prefix);
  if (get_comm().am_trainer_master()) {
    pack_scalars(p);
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

bool hypergradient_adam::load_from_checkpoint_shared(persist& p, std::string name_prefix) {
  if(p.get_cb_type() == callback_type::batch)
    optimizer::load_from_checkpoint_shared(p,name_prefix);
  struct packing_header header;
  if (get_comm().am_trainer_master()) {
    unpack_scalars(p, &header);
  }

  get_comm().trainer_broadcast(0, header);

  unpack_header(header);

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld.bin", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.read_distmat(persist_type::train, l_name, m_moment1.get());

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld.bin", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.read_distmat(persist_type::train, l_name, m_moment2.get());

  sprintf(l_name, "%s_optimizer_adam_old_gradient_%lldx%lld.bin", name_prefix.c_str(), m_old_gradient->Height(), m_old_gradient->Width());
  p.read_distmat(persist_type::train, l_name, m_old_gradient.get());
  return true;
}

bool hypergradient_adam::save_to_checkpoint_distributed(persist& p, std::string name_prefix) {
  if(p.get_cb_type() == callback_type::batch)
    optimizer::save_to_checkpoint_distributed(p,name_prefix);
  pack_scalars(p);

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_moment1);

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_moment2);

  sprintf(l_name, "%s_optimizer_adam_old_gradient_%lldx%lld", name_prefix.c_str(), m_old_gradient->Height(), m_old_gradient->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_old_gradient);

  return true;
}

bool hypergradient_adam::load_from_checkpoint_distributed(persist& p, std::string name_prefix) {
  if(p.get_cb_type() == callback_type::batch)
    optimizer::load_from_checkpoint_distributed(p,name_prefix);
  struct packing_header header;
  unpack_scalars(p, &header);

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_moment1);

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_moment2);

  sprintf(l_name, "%s_optimizer_adam_old_gradient_%lldx%lld", name_prefix.c_str(), m_old_gradient->Height(), m_old_gradient->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_old_gradient);
  return true;
}

}  // namespace lbann
