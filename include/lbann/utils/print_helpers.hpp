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

#ifndef LBANN_UTILS_PRINT_HELPERS_HPP_INCLUDED
#define LBANN_UTILS_PRINT_HELPERS_HPP_INCLUDED

#include <ostream>
#include <string>
#include <vector>

namespace lbann {

/**
 * Print a vector to an output stream.
 *
 * Requires the contained type have a valid operator<< implemented.
 */
template <typename T, typename A>
void print_vector(
  const std::vector<T, A>& v,
  std::ostream& os,
  std::string prefix = "") {
  os << prefix;
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    os << (i > 0 ? ", " : "") << v[i];
  }
  os << "]";
}

}  // namespace lbann

#endif  // LBANN_UTILS_PRINT_HELPERS_HPP_INCLUDED
