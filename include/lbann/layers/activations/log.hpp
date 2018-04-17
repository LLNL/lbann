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

#ifndef LOG_HPP_INCLUDED
#define LOG_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

#define LBANN_ENABLE_LOG_CUTOFF

namespace lbann {

/** Logarithm function. */
template <data_layout T_layout>
class log_layer : public entrywise_activation_layer {
 public:
  log_layer(lbann_comm *comm, DataType base = std::exp(1.0))
    : entrywise_activation_layer(comm),
      m_base(base),
      m_inv_log_base(1/std::log(base)),
      m_min_input(std::max(std::numeric_limits<DataType>::min(),
                           1 / std::numeric_limits<DataType>::max())) {
    if (m_base <= DataType(0)) {
      std::stringstream err;
      err << "log base (" << m_base << ") is not positive";
      LBANN_ERROR(err.str());
    }
  }
  log_layer* copy() const override { return new log_layer(*this); }
  std::string get_type() const override { return "log"; }
  data_layout get_data_layout() const override { return T_layout; }

 protected:
  DataType activation(DataType z) const override {
    if (z < m_min_input) {
      #ifdef LBANN_ENABLE_LOG_CUTOFF
      z = m_min_input;
      #else
      LBANN_ERROR("invalid input");
      #endif // LBANN_ENABLE_LOG_CUTOFF
    }
    return std::log(z) * m_inv_log_base;
  }
  DataType activation_derivative(DataType z) const override {
    if (z < m_min_input) {
      #ifdef LBANN_ENABLE_LOG_CUTOFF
      return DataType(0);
      #else
      LBANN_ERROR("invalid input");
      #endif // LBANN_ENABLE_LOG_CUTOFF
    }
    return m_inv_log_base / z;
  }

 private:

  /** Logarithm base. */
  const DataType m_base;
  /** 1 / ln(m_base). */
  const DataType m_inv_log_base;
  /** Minimum input value. */
  const DataType m_min_input;
  
};

} // namespace lbann

#endif // LOG_HPP_INCLUDED
