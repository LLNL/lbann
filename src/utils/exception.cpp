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

#include "lbann/utils/exception.hpp"
#include "lbann/utils/stack_trace.hpp"
#include "lbann/comm.hpp"

namespace lbann {
  
exception::exception(std::string message)
  : m_message(message),
    m_stack_trace(stack_trace::get()) {

  // Construct default message if none is provided
  if (m_message.empty()) {
    std::stringstream ss("LBANN exception");
    const auto& rank = get_rank_in_world();
    if (rank >= 0) {
      ss << " on rank " << rank;
    }
    m_message = ss.str();
  }
      
}

const char* exception::what() const noexcept {
  return m_message.c_str();
}

void exception::print_report(std::ostream& os) const {
  std::stringstream ss;
  ss << "****************************************************************"
     << std::endl
     << m_message << std::endl;
  if (!m_stack_trace.empty()) {
    ss << "Stack trace:" << std::endl
       << m_stack_trace;
  }
  ss << "****************************************************************"
     << std::endl;
  os << ss.str();
}
 
void exception::write(std::string base_name) const {

  // Construct file name
  std::stringstream ss(base_name);
  const auto& rank = get_rank_in_world();
  if (rank >= 0) { ss << "_rank" << rank; }
  ss << ".txt";
  const auto& file_name = ss.str();

  // Write stack trace to file
  std::ofstream fs;
  fs.open(file_name.c_str());
  print_report(fs);
  
}

} // namespace lbann
