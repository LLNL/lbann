////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/optimizers/sgd_impl.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/options.hpp"

namespace lbann {

template <typename TensorDataType>
sgd<TensorDataType>::sgd(TensorDataType learning_rate,
                         TensorDataType momentum,
                         bool nesterov)
  : BaseType(learning_rate), m_momentum(momentum), m_nesterov(nesterov)
{}

template <typename TensorDataType>
sgd<TensorDataType>::sgd(const sgd& other)
  : BaseType(other),
    m_momentum(other.m_momentum),
    m_nesterov(other.m_nesterov),
    m_velocity(other.m_velocity ? other.m_velocity->Copy() : nullptr)
{}

template <typename TensorDataType>
sgd<TensorDataType>&
sgd<TensorDataType>::operator=(const sgd<TensorDataType>& other)
{
  OptimizerType::operator=(other);
  m_momentum = other.m_momentum;
  m_nesterov = other.m_nesterov;
  m_velocity.reset(other.m_velocity ? other.m_velocity->Copy() : nullptr);
  return *this;
}

template <typename TensorDataType>
description sgd<TensorDataType>::get_description() const
{
  auto desc = OptimizerType::get_description();
  desc.add("Momentum", m_momentum);
  desc.add("Nesterov acceleration", m_nesterov);
  return desc;
}

template <typename TensorDataType>
auto sgd<TensorDataType>::get_velocity() const -> const AbsDistMatrixType&
{
  if (m_velocity == nullptr) {
    LBANN_ERROR(get_type() + " optimizer " +
                "attempted to access velocity before it was setup");
  }
  return *m_velocity;
}
template <typename TensorDataType>
auto sgd<TensorDataType>::get_velocity() -> AbsDistMatrixType&
{
  // Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers
  return const_cast<AbsDistMatrixType&>(
    static_cast<const sgd&>(*this).get_velocity());
}

template <typename TensorDataType>
void sgd<TensorDataType>::setup(WeightsType* w)
{
  OptimizerType::setup(w);
  const auto& gradient = this->get_gradient();
  m_velocity.reset(AbsDistMatrixType::Instantiate(gradient.DistData()));
#ifdef LBANN_HAS_GPU
  if (m_velocity->GetLocalDevice() == El::Device::GPU) {
    const auto& arg_parser = global_argument_parser();
    if (!arg_parser.get<bool>(
          LBANN_OPTION_USE_GPU_DEFAULT_MEMORY_IN_FORWARD_PROP)) {
      m_velocity->Matrix().SetMemoryMode(0); // Directly-allocated memory
    }
  }
#endif // LBANN_HAS_GPU
  El::Zeros(*m_velocity, gradient.Height(), gradient.Width());
}

template <typename TensorDataType>
void sgd<TensorDataType>::write_proto(lbann_data::Optimizer& proto) const
{
  auto* opt = proto.mutable_sgd();
  opt->set_learn_rate(this->get_learning_rate());
  opt->set_momentum(m_momentum);
  opt->set_nesterov(m_nesterov);
}

template <typename TensorDataType>
void sgd<TensorDataType>::step_compute(AbsDistMatrixType& values,
                                       const AbsDistMatrixType& gradient)
{
  if (m_momentum == TensorDataType(0.)) {
    // Vanilla SGD
    El::Axpy(-this->get_learning_rate(), gradient, values);
  }
  else {
    // Momentum or Nesterov SGD
    switch (values.GetLocalDevice()) {
    case El::Device::CPU:
      momentum_step_cpu(values, gradient);
      break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      momentum_step_gpu(values, gradient);
      break;
#endif // LBANN_HAS_GPU
    default:
      std::ostringstream err;
      err << "unsupported device type "
          << "(" << static_cast<int>(values.GetLocalDevice()) << ")";
      LBANN_ERROR(err.str());
    }
  }
}

template <typename TensorDataType>
void sgd<TensorDataType>::momentum_step_cpu(AbsDistMatrixType& values,
                                            const AbsDistMatrixType& gradient)
{
  LBANN_CALIPER_MARK_SCOPE("sgd::momentum_step");
  // Get local matrix data
  const auto learning_rate = El::To<TensorDataType>(this->get_learning_rate());
  const size_t local_height = values.LocalHeight();
  const size_t local_width = values.LocalWidth();
  auto* __restrict__ values_buffer = values.Buffer();
  const auto* __restrict__ gradient_buffer = gradient.LockedBuffer();
  auto* __restrict__ velocity_buffer = m_velocity->Buffer();

  if (values.Contiguous() && gradient.Contiguous() &&
      m_velocity->Contiguous()) {
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
    }
    else {

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
  }
  else {

    // Momentum or Nesterov SGD with non-contiguous data
    const size_t values_ldim = values.LDim();
    const size_t gradient_ldim = gradient.LDim();
    const size_t velocity_ldim = m_velocity->LDim();
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t col = 0; col < local_width; ++col) {
      for (size_t row = 0; row < local_height; ++row) {
        const auto& g = gradient_buffer[row + col * gradient_ldim];
        auto& v = velocity_buffer[row + col * velocity_ldim];
        auto& x = values_buffer[row + col * values_ldim];
        v = m_momentum * v + g;
        x -= (m_nesterov ? learning_rate * (m_momentum * v + g)
                         : learning_rate * v);
      }
    }
  }
}

template <typename TensorDataType>
std::unique_ptr<optimizer>
build_sgd_optimizer_from_pbuf(google::protobuf::Message const& msg)
{
  const auto& params = dynamic_cast<lbann_data::Optimizer::SGD const&>(msg);
  return std::make_unique<sgd<TensorDataType>>(
    TensorDataType(params.learn_rate()),
    TensorDataType(params.momentum()),
    params.nesterov());
}

#define PROTO(T)                                                               \
  template class sgd<T>;                                                       \
  template std::unique_ptr<optimizer> build_sgd_optimizer_from_pbuf<T>(        \
    google::protobuf::Message const&)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
} // namespace lbann

#define LBANN_CLASS_NAME sgd
#include <lbann/macros/register_template_class_with_cereal.hpp>
