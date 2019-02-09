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

#include "lbann/optimizers/adam.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

adam::adam(lbann_comm *comm,
           DataType learning_rate,
           DataType beta1,
           DataType beta2,
           DataType eps)
  : optimizer(comm, learning_rate),
    m_beta1(beta1), m_beta2(beta2), m_eps(eps) {}

adam::adam(const adam& other)
  : optimizer(other),
    m_beta1(other.m_beta1),
    m_beta2(other.m_beta2),
    m_eps(other.m_eps),
    m_current_beta1(other.m_current_beta1),
    m_current_beta2(other.m_current_beta2),
    m_moment1(other.m_moment1 ? other.m_moment1->Copy() : nullptr),
    m_moment2(other.m_moment2 ? other.m_moment2->Copy() : nullptr) {}

adam& adam::operator=(const adam& other) {
  optimizer::operator=(other);
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

description adam::get_description() const {
  auto&& desc = optimizer::get_description();
  desc.add("beta1", m_beta1);
  desc.add("beta2", m_beta2);
  desc.add("eps", m_eps);
  return desc;
}

const AbsDistMat& adam::get_moment1() const {
  if (m_moment1 == nullptr) {
    LBANN_ERROR(get_type() + " optimizer "
                + "attempted to access moment1 before it was setup");
  }
  return *m_moment1;
}
AbsDistMat& adam::get_moment1() {
  // Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers
  return const_cast<AbsDistMat&>(static_cast<const adam&>(*this).get_moment1());
}
const AbsDistMat& adam::get_moment2() const {
  if (m_moment2 == nullptr) {
    LBANN_ERROR(get_type() + " optimizer "
                + "attempted to access moment2 before it was setup");
  }
  return *m_moment2;
}
AbsDistMat& adam::get_moment2() {
  // Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers
  return const_cast<AbsDistMat&>(static_cast<const adam&>(*this).get_moment2());
}

void adam::setup(weights& w) {
  optimizer::setup(w);
  const auto& gradient = this->get_gradient();
  m_moment1.reset(AbsDistMat::Instantiate(gradient.DistData()));
  m_moment2.reset(AbsDistMat::Instantiate(gradient.DistData()));
  El::Zeros(*m_moment1, gradient.Height(), gradient.Width());
  El::Zeros(*m_moment2, gradient.Height(), gradient.Width());
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
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
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
    LBANN_OMP_PARALLEL_FOR
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

////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

bool adam::save_to_checkpoint_shared(persist& p, std::string name_prefix) {
  optimizer::save_to_checkpoint_shared(p, name_prefix);

  if (m_comm->am_trainer_master()) {
    pack_scalars(p);
  }

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.write_distmat(persist_type::train, l_name, m_moment1.get());

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.write_distmat(persist_type::train, l_name, m_moment2.get());

  return true;
}

bool adam::load_from_checkpoint_shared(persist& p, std::string name_prefix) {
  optimizer::load_from_checkpoint_shared(p, name_prefix);
  struct packing_header header;
  if (m_comm->am_trainer_master()) {
    unpack_scalars(p, &header);
  }

  m_comm->trainer_broadcast(0, header);

  unpack_header(header);

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld.bin", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.read_distmat(persist_type::train, l_name, m_moment1.get());

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld.bin", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.read_distmat(persist_type::train, l_name, m_moment2.get());

  return true;
}

bool adam::save_to_checkpoint_distributed(persist& p, std::string name_prefix) {
  optimizer::save_to_checkpoint_distributed(p, name_prefix);

  pack_scalars(p);

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_moment1);

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.write_rank_distmat(persist_type::train, l_name, *m_moment2);

  return true;
}

bool adam::load_from_checkpoint_distributed(persist& p, std::string name_prefix) {
  optimizer::load_from_checkpoint_distributed(p, name_prefix);
  struct packing_header header;
  unpack_scalars(p, &header);

  char l_name[512];
  sprintf(l_name, "%s_optimizer_adam_moment1_%lldx%lld", name_prefix.c_str(), m_moment1->Height(), m_moment2->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_moment1);

  sprintf(l_name, "%s_optimizer_adam_moment2_%lldx%lld", name_prefix.c_str(), m_moment2->Height(), m_moment2->Width());
  p.read_rank_distmat(persist_type::train, l_name, *m_moment2);

  return true;
}

} // namespace lbann
