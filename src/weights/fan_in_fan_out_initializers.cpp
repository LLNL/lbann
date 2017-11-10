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
//
// fan_in_fan_out_initializer .hpp .cpp - Fan-in-fan-out initializer classes
////////////////////////////////////////////////////////////////////////////////

#include "lbann/weights/fan_in_fan_out_initializers.hpp"

namespace lbann {

void glorot_normal_initializer::intialize_entries(AbsDistMat& weights_matrix) const {
  if (m_fan_in <= 0 || m_fan_out <= 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid dimensions for Glorot normal initialization "
        << "(fan_in=" << this->m_fan_in << ",fan_out=" << this->m_fan_out << ")";
    throw lbann_exception(err.str());
  }
  const DataType variance = DataType(2) / (m_fan_in + m_fan_out);
  gaussian_fill(weights_matrix,
                height, width,
                DataType(0), std::sqrt(variance));
}

void glorot_uniform_initializer::intialize_entries(AbsDistMat& weights_matrix) const {
  if (m_fan_in <= 0 || m_fan_out <= 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid dimensions for Glorot uniform initialization "
        << "(fan_in=" << this->m_fan_in << ",fan_out=" << this->m_fan_out << ")";
    throw lbann_exception(err.str());
  }
  const DataType variance = DataType(2) / (m_fan_in + m_fan_out);
  gaussian_fill(weights_matrix,
                height, width,
                DataType(0), std::sqrt(3*variance));
}

void he_normal_initializer::intialize_entries(AbsDistMat& weights_matrix) const {
  if (m_fan_in <= 0 || m_fan_out <= 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid dimensions for He normal initialization "
        << "(fan_in=" << this->m_fan_in << ",fan_out=" << this->m_fan_out << ")";
    throw lbann_exception(err.str());
  }
  const DataType variance = DataType(1) / m_fan_in;
  gaussian_fill(weights_matrix,
                height, width,
                DataType(0), std::sqrt(variance));
}

void he_uniform_initializer::intialize_entries(AbsDistMat& weights_matrix) const {
  if (m_fan_in <= 0 || m_fan_out <= 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid dimensions for He uniform initialization "
        << "(fan_in=" << this->m_fan_in << ",fan_out=" << this->m_fan_out << ")";
    throw lbann_exception(err.str());
  }
  const DataType variance = DataType(1) / m_fan_in;
  gaussian_fill(weights_matrix,
                height, width,
                DataType(0), std::sqrt(3*variance));
}

}  // namespace lbann
