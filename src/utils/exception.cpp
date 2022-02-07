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

#include "lbann/utils/exception.hpp"

#include "lbann/comm_impl.hpp"
#include "lbann/utils/stack_trace.hpp"

namespace lbann {

static std::string default_error_message()
{
  std::ostringstream oss("LBANN exception", std::ios_base::ate);
  const auto rank = get_rank_in_world();
  if (rank >= 0) {
    oss << " on rank " << rank;
  }
  return oss.str();
}

exception::exception() : exception(default_error_message()) {}

exception::exception(std::string message)
{
  // Build the "real" what() string, complete with stack trace:
  auto const stack_trace = stack_trace::get();
  auto const star_rule = std::string(64, '*');
  std::ostringstream oss;
  oss << star_rule << "\n" << message;
  if (!stack_trace.empty()) {
    oss << "Stack trace:\n" << stack_trace;
  }
  oss << star_rule << "\n";
  m_message = oss.str();
}

const char* exception::what() const noexcept { return m_message.c_str(); }

void exception::print_report(std::ostream& os) const { os << m_message; }

} // namespace lbann
