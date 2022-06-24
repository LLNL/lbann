////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#define LBANN_VARIANCE_SCALING_INITIALIZER_INSTANTIATE
#include "lbann/weights/variance_scaling_initializers.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

#include <weights.pb.h>

namespace lbann {
namespace {
using ValidDataTypes = h2::meta::TL<
#ifdef LBANN_HAS_GPU_FP16
  fp16,
#endif
#ifdef LBANN_HAS_HALF
  cpu_fp16,
#endif
  float, double>;

using InitTypes =
  h2::meta::tlist::ExpandTL<variance_scaling_initializer, ValidDataTypes>;

struct default_errors {
  template <typename... Ts>
  void DispatchError(Ts&&...) {
    LBANN_ERROR("Failed to dispatch.");
  }
  template <typename... Ts>
  void DeductionError(Ts&&...) {
    // In this case, the initializer is just not a variance scaling
    // initializer. This isn't a problem, so do nothing.
  }
};

struct fan_in_functor : default_errors {
  fan_in_functor(double val) : value_{val} {}
  template <typename DType>
  void operator()(variance_scaling_initializer<DType>& init)
  {
    init.set_fan_in(value_);
  }
  double value_;
};
struct fan_out_functor : default_errors {
  fan_out_functor(double val) : value_{val} {}
  template <typename DType>
  void operator()(variance_scaling_initializer<DType>& init)
  {
    init.set_fan_out(value_);
  }
  double value_;
};
}

void set_fan_in(weights_initializer& initializer, double value) {
  using Dispatcher =
    h2::multimethods::SwitchDispatcher<fan_in_functor,
                                       void,
                                       weights_initializer, InitTypes>;
  Dispatcher::Exec(fan_in_functor(value), initializer);
}

void set_fan_out(weights_initializer& initializer, double value) {
  using Dispatcher =
    h2::multimethods::SwitchDispatcher<fan_out_functor,
                                       void,
                                       weights_initializer, InitTypes>;
  Dispatcher::Exec(fan_out_functor(value), initializer);
}

template <typename TensorDataType>
variance_scaling_initializer<TensorDataType>::variance_scaling_initializer(probability_distribution dist)
  : m_prob_dist(dist),
    m_fan_in(0),
    m_fan_out(0) {
  if (m_prob_dist != probability_distribution::gaussian
      && m_prob_dist != probability_distribution::uniform) {
    std::stringstream err;
    err << "variance scaling initializer is only supported with "
        << "Gaussian and uniform probability distributions "
        << "(dist=" << int(m_prob_dist) << ")";
    LBANN_ERROR(err.str());
  }
}

template <typename TensorDataType>
description variance_scaling_initializer<TensorDataType>::get_description() const {
  auto desc = data_type_weights_initializer<TensorDataType>::get_description();
  std::string dist_str;
  switch (m_prob_dist) {
  case probability_distribution::gaussian:
    dist_str = "normal";  break;
  case probability_distribution::uniform:
    dist_str = "uniform"; break;
  default:
    dist_str = "invalid";
  }
  desc.add("Distribution", dist_str);
  desc.add("Fan-in", m_fan_in);
  desc.add("Fan-out", m_fan_out);
  return desc;
}

template <typename TensorDataType>
void variance_scaling_initializer<TensorDataType>::fill(El::AbstractDistMatrix<TensorDataType>& matrix) {

  // Check if fan-in and fan-out parameters are valid
  if (m_fan_in <= 0 || m_fan_out <= 0) {
    std::stringstream err;
    err << "attempted variance scaling initialization "
        << "with invalid parameters "
        << "(fan_in=" << m_fan_in << ",fan_out=" << m_fan_out << ")";
    LBANN_ERROR(err.str());
  }

  // Get variance
  const auto& variance = get_variance(m_fan_in, m_fan_out);

  // Fill matrix with values drawn from probability distribution
  switch (m_prob_dist) {
  case probability_distribution::gaussian:
    gaussian_fill(matrix, matrix.Height(), matrix.Width(),
                  TensorDataType(0.), El::Sqrt(variance));
    break;
  case probability_distribution::uniform:
    uniform_fill(matrix, matrix.Height(), matrix.Width(),
                 TensorDataType(0.), El::Sqrt(El::To<TensorDataType>(3)*variance));
    break;
  default:
    std::stringstream err;
    err << "variance scaling initializer is only supported with "
        << "Gaussian and uniform probability distributions "
        << "(dist=" << El::Int(m_prob_dist) << ")";
    LBANN_ERROR(err.str());
  }

}

template <typename TensorDataType>
TensorDataType glorot_initializer<TensorDataType>::get_variance(El::Int fan_in, El::Int fan_out) {
  return El::To<TensorDataType>(2.0) / El::To<TensorDataType>(fan_in + fan_out);
}

template <typename TensorDataType>
void glorot_initializer<TensorDataType>::write_specific_proto(
  lbann_data::Initializer& init) const
{
  if (this->get_prob_dist() == probability_distribution::uniform)
    init.mutable_glorot_uniform_initializer();
  else
    init.mutable_glorot_normal_initializer();
}

template <typename TensorDataType>
TensorDataType he_initializer<TensorDataType>::get_variance(El::Int fan_in, El::Int fan_out) {
  return El::To<TensorDataType>(2.0) / El::To<TensorDataType>(fan_in);
}

template <typename TensorDataType>
void he_initializer<TensorDataType>::write_specific_proto(
  lbann_data::Initializer& init) const
{
  if (this->get_prob_dist() == probability_distribution::uniform)
    init.mutable_he_uniform_initializer();
  else
    init.mutable_he_normal_initializer();
}

template <typename TensorDataType>
TensorDataType lecun_initializer<TensorDataType>::get_variance(El::Int fan_in, El::Int fan_out) {
  return El::TypeTraits<TensorDataType>::One() / El::To<TensorDataType>(fan_in);
}

template <typename TensorDataType>
void lecun_initializer<TensorDataType>::write_specific_proto(
  lbann_data::Initializer& init) const
{
  if (this->get_prob_dist() == probability_distribution::uniform)
    init.mutable_lecun_uniform_initializer();
  else
    init.mutable_lecun_normal_initializer();
}

//
// Builder functions
//

// FIXME (trb 07/31/2019): This is kinda ugly, but its fine if there
// are only 2 probability distributions
template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_glorot_initializer_from_pbuf(google::protobuf::Message const& msg) {
  if (dynamic_cast<lbann_data::Initializer::GlorotNormalInitializer const*>(&msg))
    return std::make_unique<glorot_initializer<TensorDataType>>(probability_distribution::gaussian);
  else if (dynamic_cast<lbann_data::Initializer::GlorotUniformInitializer const*>(&msg))
    return std::make_unique<glorot_initializer<TensorDataType>>(probability_distribution::uniform);
  else {
    LBANN_ERROR("build_glorot_initializer_from_pbuf: Bad message.");
    return nullptr;
  }
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_he_initializer_from_pbuf(google::protobuf::Message const& msg) {
  if (dynamic_cast<lbann_data::Initializer::HeNormalInitializer const*>(&msg))
    return std::make_unique<he_initializer<TensorDataType>>(probability_distribution::gaussian);
  else if (dynamic_cast<lbann_data::Initializer::HeUniformInitializer const*>(&msg))
    return std::make_unique<he_initializer<TensorDataType>>(probability_distribution::uniform);
  else {
    LBANN_ERROR("build_he_initializer_from_pbuf: Bad message.");
    return nullptr;
  }
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_lecun_initializer_from_pbuf(google::protobuf::Message const& msg) {
  if (dynamic_cast<lbann_data::Initializer::LeCunNormalInitializer const*>(&msg))
    return std::make_unique<lecun_initializer<TensorDataType>>(probability_distribution::gaussian);
  else if (dynamic_cast<lbann_data::Initializer::LeCunUniformInitializer const*>(&msg))
    return std::make_unique<lecun_initializer<TensorDataType>>(probability_distribution::uniform);
  else {
    LBANN_ERROR("build_lecun_initializer_from_pbuf: Bad message.");
    return nullptr;
  }
}


#define PROTO(T)                                                           \
  template class glorot_initializer<T>;                                    \
  template class he_initializer<T>;                                        \
  template class lecun_initializer<T>;                                     \
  template std::unique_ptr<weights_initializer>                            \
  build_glorot_initializer_from_pbuf<T>(google::protobuf::Message const&); \
  template std::unique_ptr<weights_initializer>                            \
  build_he_initializer_from_pbuf<T>(google::protobuf::Message const&);     \
  template std::unique_ptr<weights_initializer>                            \
  build_lecun_initializer_from_pbuf<T>(google::protobuf::Message const&);  \

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace lbann
