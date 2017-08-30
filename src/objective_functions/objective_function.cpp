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

#include "lbann/objective_functions/objective_function.hpp"

namespace lbann {

namespace objective_functions {

objective_function::objective_function() {
  reset_statistics();
}

void objective_function::add_to_value(double value) {
  m_value += value;
}

void objective_function::record_and_reset_value() {
  m_recorded_values += m_value;
  m_recorded_iterations++;
  m_value = 0.0;
}

double objective_function::get_value() const {
  return m_value;
}

double objective_function::get_mean_value() const {
  if(m_recorded_iterations != 0) {
    return m_recorded_values / m_recorded_iterations;
  }
  else {
    return std::nan("");
  }
}

void objective_function::reset_statistics() {
  m_value = 0.0;
  m_recorded_values = 0.0;
  m_recorded_iterations = 0;
}

}  // namespace objective_functions

}  // namespace lbann
