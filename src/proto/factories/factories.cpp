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

#include "lbann/proto/factories.hpp"

namespace lbann {
namespace proto {

/** Parse a space-separated list of execution modes. */
template <>
std::vector<execution_mode> parse_list<execution_mode>(std::string str) {
  std::vector<execution_mode> list;
  for (const auto& mode : parse_list<std::string>(str)) {
    if (mode == "train" || mode == "training") {
      list.push_back(execution_mode::training);
    } else if (mode == "validate" || mode == "validation") {
      list.push_back(execution_mode::validation);
    } else if (mode == "test" || mode == "testing") {
      list.push_back(execution_mode::testing);
    } else {
      LBANN_ERROR("invalid execution mode (\"" + mode + "\")");
    }
  }
  return list;
}

} // namespace proto
} // namespace lbann
