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

#define LBANN_VARIANCE_SCALING_INITIALIZER_INSTANTIATE
#include "lbann/weights/variance_scaling_initializers.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"

#include <weights.pb.h>

namespace lbann {

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
TensorDataType he_initializer<TensorDataType>::get_variance(El::Int fan_in, El::Int fan_out) {
  return El::To<TensorDataType>(2.0) / El::To<TensorDataType>(fan_in);
}

template <typename TensorDataType>
TensorDataType lecun_initializer<TensorDataType>::get_variance(El::Int fan_in, El::Int fan_out) {
  return El::TypeTraits<TensorDataType>::One() / El::To<TensorDataType>(fan_in);
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
    return make_unique<glorot_initializer<TensorDataType>>(probability_distribution::gaussian);
  else if (dynamic_cast<lbann_data::Initializer::GlorotUniformInitializer const*>(&msg))
    return make_unique<glorot_initializer<TensorDataType>>(probability_distribution::uniform);
  else {
    LBANN_ERROR("build_glorot_initializer_from_pbuf: Bad message.");
    return nullptr;
  }
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_he_initializer_from_pbuf(google::protobuf::Message const& msg) {
  if (dynamic_cast<lbann_data::Initializer::HeNormalInitializer const*>(&msg))
    return make_unique<he_initializer<TensorDataType>>(probability_distribution::gaussian);
  else if (dynamic_cast<lbann_data::Initializer::HeUniformInitializer const*>(&msg))
    return make_unique<he_initializer<TensorDataType>>(probability_distribution::uniform);
  else {
    LBANN_ERROR("build_he_initializer_from_pbuf: Bad message.");
    return nullptr;
  }
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_lecun_initializer_from_pbuf(google::protobuf::Message const& msg) {
  if (dynamic_cast<lbann_data::Initializer::LeCunNormalInitializer const*>(&msg))
    return make_unique<lecun_initializer<TensorDataType>>(probability_distribution::gaussian);
  else if (dynamic_cast<lbann_data::Initializer::LeCunUniformInitializer const*>(&msg))
    return make_unique<lecun_initializer<TensorDataType>>(probability_distribution::uniform);
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
