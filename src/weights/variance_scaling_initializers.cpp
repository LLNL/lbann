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

#include "lbann/weights/variance_scaling_initializers.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

variance_scaling_initializer::variance_scaling_initializer(probability_distribution dist)
  : weights_initializer(),
    m_prob_dist(dist),
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

description variance_scaling_initializer::get_description() const {
  auto desc = weights_initializer::get_description();
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

void variance_scaling_initializer::fill(AbsDistMat& matrix) {

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
                  DataType(0), std::sqrt(variance));
    break;
  case probability_distribution::uniform:
    uniform_fill(matrix, matrix.Height(), matrix.Width(),
                 DataType(0), std::sqrt(3*variance));
    break;
  default:
    std::stringstream err;
    err << "variance scaling initializer is only supported with "
        << "Gaussian and uniform probability distributions "
        << "(dist=" << El::Int(m_prob_dist) << ")";
    LBANN_ERROR(err.str());
  }

}

DataType glorot_initializer::get_variance(El::Int fan_in, El::Int fan_out) {
  return DataType(2) / (fan_in + fan_out);
}

DataType he_initializer::get_variance(El::Int fan_in, El::Int fan_out) {
  return DataType(2) / fan_in;
}

DataType lecun_initializer::get_variance(El::Int fan_in, El::Int fan_out) {
  return DataType(1) / fan_in;
}

}  // namespace lbann
