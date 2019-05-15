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

#include "lbann/optimizers/sgd.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

sgd::sgd(lbann_comm* comm,
         DataType learning_rate,
         DataType momentum,
         bool nesterov)
  : optimizer(comm, learning_rate),
    m_momentum(momentum),
    m_nesterov(nesterov) {}

sgd::sgd(const sgd& other)
  : optimizer(other),
    m_momentum(other.m_momentum),
    m_nesterov(other.m_nesterov),
    m_velocity(other.m_velocity ? other.m_velocity->Copy() : nullptr) {}

sgd& sgd::operator=(const sgd& other) {
  optimizer::operator=(other);
  m_momentum = other.m_momentum;
  m_nesterov = other.m_nesterov;
  m_velocity.reset(other.m_velocity ?
                   other.m_velocity->Copy() : nullptr);
  return *this;
}

description sgd::get_description() const {
  auto&& desc = optimizer::get_description();
  desc.add("Momentum", m_momentum);
  desc.add("Nesterov acceleration", m_nesterov);
  return desc;
}

const AbsDistMat& sgd::get_velocity() const {
  if (m_velocity == nullptr) {
    LBANN_ERROR(get_type() + " optimizer "
                + "attempted to access velocity before it was setup");
  }
  return *m_velocity;
}
AbsDistMat& sgd::get_velocity() {
  // Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers
  return const_cast<AbsDistMat&>(static_cast<const sgd&>(*this).get_velocity());
}

void sgd::setup(weights* w) {
  optimizer::setup(w);
  const auto& gradient = this->get_gradient();
  m_velocity.reset(AbsDistMat::Instantiate(gradient.DistData()));
  El::Zeros(*m_velocity, gradient.Height(), gradient.Width());
}

void sgd::step_compute(AbsDistMat& values, const AbsDistMat& gradient) {
  if (m_momentum == DataType(0)) {
    // Vanilla SGD
    El::Axpy(-this->get_learning_rate(), gradient, values);
  } else {
    // Momentum or Nesterov SGD
    switch (values.GetLocalDevice()) {
    case El::Device::CPU: momentum_step_cpu(values, gradient); break;
#ifdef LBANN_HAS_CUDA
    case El::Device::GPU: momentum_step_gpu(values, gradient); break;
#endif // LBANN_HAS_CUDA
    default:
      std::ostringstream err;
      err << "unsupported device type "
        << "(" << static_cast<int>(values.GetLocalDevice()) << ")";
      LBANN_ERROR(err.str());
    }
  }
}

void sgd::momentum_step_cpu(AbsDistMat& values, const AbsDistMat& gradient) {

  // Get local matrix data
  const auto& learning_rate = this->get_learning_rate();
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  auto* __restrict__ values_buffer = values.Buffer();
  const auto* __restrict__ gradient_buffer = gradient.LockedBuffer();
  auto* __restrict__ velocity_buffer = m_velocity->Buffer();

  if (values.Contiguous() && gradient.Contiguous()
      && m_velocity->Contiguous()) {
    const size_t local_size = local_height * local_width;
    if (m_nesterov) {

      // Nesterov SGD for contiguous data
      LBANN_OMP_PARALLEL_FOR
      for (size_t i = 0; i < local_size; ++i) {
        auto& x = values_buffer[i];
        const auto& g = gradient_buffer[i];
        auto& v = velocity_buffer[i];
        v = m_momentum * v + g;
        x -= learning_rate * (m_momentum * v + g);
      }

    } else {

      // Momentum SGD with contiguous data
      LBANN_OMP_PARALLEL_FOR
      for (size_t i = 0; i < local_size; ++i) {
        auto& x = values_buffer[i];
        const auto& g = gradient_buffer[i];
        auto& v = velocity_buffer[i];
        v = m_momentum * v + g;
        x -= learning_rate * v;
      }

    }
  } else {

    // Momentum or Nesterov SGD with non-contiguous data
    const size_t values_ldim = values.LDim();
    const size_t gradient_ldim = gradient.LDim();
    const size_t velocity_ldim = m_velocity->LDim();
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t col = 0; col < local_width; ++col) {
      for (size_t row=0; row < local_height; ++row) {
        const auto& g = gradient_buffer[row+col*gradient_ldim];
        auto& v = velocity_buffer[row+col*velocity_ldim];
        auto& x = values_buffer[row+col*values_ldim];
        v = m_momentum * v + g;
        x -= (m_nesterov ?
              learning_rate * (m_momentum * v + g) :
              learning_rate * v);
      }
    }

  }

}

// =============================================
// Checkpointing
// =============================================

bool sgd::save_to_checkpoint_shared(persist& p, std::string name_prefix) {
  optimizer::save_to_checkpoint_shared(p, name_prefix);

  if (get_comm().am_trainer_master()) {
    pack_scalars(p);
  }

  char l_name[512];
  sprintf(l_name, "%s_optimizer_velocity_%lldx%lld", name_prefix.c_str(), m_velocity->Height(), m_velocity->Width());
  p.write_distmat(persist_type::train, l_name, m_velocity.get());

  return true;
}

bool sgd::load_from_checkpoint_shared(persist& p, std::string name_prefix) {
  optimizer::load_from_checkpoint_shared(p, name_prefix);
  struct packing_header header;
  if (get_comm().am_trainer_master()) {
    unpack_scalars(p, &header);
  }

  get_comm().trainer_broadcast(0, header);

  unpack_header(header);
  char l_name[512];
  sprintf(l_name, "%s_optimizer_velocity_%lldx%lld.bin", name_prefix.c_str(), m_velocity->Height(), m_velocity->Width());
  p.read_distmat(persist_type::train, l_name, m_velocity.get());

  return true;
}

bool sgd::save_to_checkpoint_distributed(persist& p, std::string name_prefix) {
  optimizer::save_to_checkpoint_distributed(p, name_prefix);

  pack_scalars(p);

  char l_name[512];
  sprintf(l_name, "%s_optimizer_velocity_%lldx%lld", name_prefix.c_str(), m_velocity->LocalHeight(), m_velocity->LocalWidth());
  p.write_rank_distmat(persist_type::train, l_name, *m_velocity);

  return true;
}

bool sgd::load_from_checkpoint_distributed(persist& p, std::string name_prefix) {
  optimizer::load_from_checkpoint_distributed(p, name_prefix);
  struct packing_header header;
  unpack_scalars(p, &header);

  char l_name[512];
  sprintf(l_name, "%s_optimizer_velocity_%lldx%lld", name_prefix.c_str(), m_velocity->LocalHeight(), m_velocity->LocalWidth());
  p.read_rank_distmat(persist_type::train, l_name, *m_velocity);

  return true;
}

} // namespace lbann
